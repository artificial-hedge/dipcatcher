"""KATs for managed API keys — ApiKeyStore, /harness/keys routes, auth wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore, KeyStoreError


class _B:
    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


def _client(env_key: str | None = "root-secret", **kw: Any) -> TestClient:
    import os

    if env_key is None:
        os.environ.pop("FX1_API_KEY", None)
    else:
        os.environ["FX1_API_KEY"] = env_key
    return TestClient(
        create_app(backend_resolver=lambda *a, **k: _B(), **kw),
        raise_server_exceptions=False,
    )


@pytest.fixture(autouse=True)
def _clean_env():
    import os

    saved = os.environ.get("FX1_API_KEY")
    yield
    if saved is None:
        os.environ.pop("FX1_API_KEY", None)
    else:
        os.environ["FX1_API_KEY"] = saved


# ---- store ---------------------------------------------------------------


def test_store_mint_returns_raw_once_and_never_stores_it() -> None:
    store = ApiKeyStore()
    raw, rec = store.mint("svc-a")
    assert raw.startswith("fx1k_")
    assert rec["key_id"] and rec["prefix"] == raw[:13]
    assert "sha256" not in rec
    # the secret never appears in a list/get view either
    listed = store.list()
    assert len(listed) == 1
    assert raw not in str(listed)
    assert "sha256" not in str(listed)


def test_store_authenticate_counts_uses() -> None:
    store = ApiKeyStore()
    raw, rec = store.mint()
    first = store.authenticate(raw)
    assert first is not None and first["key_id"] == rec["key_id"]
    again = store.authenticate(raw)
    assert again is not None and again["uses"] == 2
    assert again["last_used_at"] is not None


def test_store_authenticate_rejects_garbage_and_revoked() -> None:
    store = ApiKeyStore()
    raw, rec = store.mint()
    assert store.authenticate("not-a-key") is None
    assert store.authenticate("fx1k_" + "0" * 40) is None
    out = store.revoke(rec["key_id"])
    assert out["enabled"] is False and out["revoked_at"] is not None
    assert store.authenticate(raw) is None


def test_store_revoke_fails_closed() -> None:
    store = ApiKeyStore()
    with pytest.raises(KeyStoreError):
        store.revoke("deadbeefdeadbeef")
    _, rec = store.mint()
    store.revoke(rec["key_id"])
    with pytest.raises(KeyStoreError):
        store.revoke(rec["key_id"])


def test_store_cap_refuses() -> None:
    store = ApiKeyStore(max_keys=2)
    store.mint()
    store.mint()
    with pytest.raises(KeyStoreError):
        store.mint()


def test_store_journal_recovers_mint_and_revoke(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path))
    raw_a, _ = store.mint("a")
    _, rec_b = store.mint("b")
    store.revoke(rec_b["key_id"])
    # fresh store over the same journal replays the same state
    restored = ApiKeyStore(journal=JobJournal(path))
    assert restored.authenticate(raw_a) is not None
    assert restored.get(rec_b["key_id"]) is not None
    assert restored.get(rec_b["key_id"])["enabled"] is False


# ---- wire -----------------------------------------------------------------


def test_managed_key_authenticates_and_is_not_admin() -> None:
    client = _client()
    root = {"X-API-Key": "root-secret"}
    mint = client.post("/harness/keys", json={"name": "svc"}, headers=root)
    assert mint.status_code == 201
    body = mint.json()
    raw = body["key"]
    kid = body["id"]
    # managed key serves gated routes
    ok = client.get("/harness/commands", headers={"X-API-Key": raw})
    assert ok.status_code == 200
    # but key management is admin-only
    denied = client.get("/harness/keys", headers={"X-API-Key": raw})
    assert denied.status_code == 403
    assert denied.json()["code"] == "admin_required"
    # env key lists it
    listed = client.get("/harness/keys", headers=root)
    assert listed.status_code == 200
    ids = [k["id"] for k in listed.json()["data"]]
    assert ids == [kid]
    assert raw not in str(listed.json())
    # revoked → auth fails closed
    gone = client.delete(f"/harness/keys/{kid}", headers=root)
    assert gone.status_code == 200 and gone.json()["enabled"] is False
    assert client.get("/harness/commands", headers={"X-API-Key": raw}).status_code == 401


def test_key_admin_requires_bootstrap() -> None:
    client = _client()
    mint = client.post("/harness/keys", json={}, headers={"X-API-Key": "root-secret"})
    raw = mint.json()["key"]
    assert mint.json()["admin"] is False
    # managed key cannot mint more keys
    assert client.post("/harness/keys", json={}, headers={"X-API-Key": raw}).status_code == 403
    # no credential cannot either
    assert client.post("/harness/keys", json={}).status_code == 401


def test_admin_key_manages_keys() -> None:
    """An admin managed key can mint/list/revoke — so a deployment with
    no env key keeps a control plane after provisioning turns auth on."""
    client = _client()
    mint = client.post(
        "/harness/keys",
        json={"name": "ops", "admin": True},
        headers={"X-API-Key": "root-secret"},
    )
    assert mint.status_code == 201 and mint.json()["admin"] is True
    ahead = {"X-API-Key": mint.json()["key"]}
    assert client.post("/harness/keys", json={}, headers=ahead).status_code == 201
    assert client.get("/harness/keys", headers=ahead).status_code == 200
    listed = client.get("/harness/keys", headers=ahead).json()["data"]
    admin_rec = next(k for k in listed if k["admin"])
    gone = client.delete(f"/harness/keys/{admin_rec['id']}", headers=ahead)
    assert gone.status_code == 200
    # the admin key revoked itself → its own auth now fails closed
    assert client.get("/harness/keys", headers=ahead).status_code == 401


def test_key_get_404_and_revoke_409() -> None:
    client = _client()
    root = {"X-API-Key": "root-secret"}
    assert client.get("/harness/keys/nope", headers=root).status_code == 404
    kid = client.post("/harness/keys", json={}, headers=root).json()["id"]
    client.delete(f"/harness/keys/{kid}", headers=root)
    assert client.delete(f"/harness/keys/{kid}", headers=root).status_code == 409


def test_completion_records_attribute_the_key() -> None:
    client = _client()
    root = {"X-API-Key": "root-secret"}
    mint = client.post("/harness/keys", json={}, headers=root)
    raw, kid = mint.json()["key"], mint.json()["id"]
    res = client.post(
        "/harness/complete",
        json={"messages": [{"role": "user", "content": "hi"}], "backend": "hosted_k3"},
        headers={"X-API-Key": raw},
    )
    assert res.status_code == 200
    usage = client.get(f"/harness/usage?key_id={kid}", headers=root).json()
    assert usage["records_seen"] == 1
    assert usage["by_key"][kid]["requests"] == 1
    # unfiltered view splits managed-key calls from env-key calls
    all_usage = client.get("/harness/usage", headers=root).json()
    assert all_usage["by_key"][kid]["requests"] == 1


def test_keys_bootstrap_remote_auth_without_env_key() -> None:
    # no env key, empty store: loopback dev — mint is allowed
    client = _client(env_key=None)
    mint = client.post("/harness/keys", json={"admin": True})
    assert mint.status_code == 201
    raw = mint.json()["key"]
    # minting turned auth on — a bare request now 401s even on loopback
    assert client.get("/harness/commands").status_code == 401
    # the admin managed key authenticates and keeps the control plane
    assert client.get("/harness/commands", headers={"X-API-Key": raw}).status_code == 200
    assert client.get("/harness/keys", headers={"X-API-Key": raw}).status_code == 200


def test_keys_journal_under_state_dir(tmp_path: Path) -> None:
    client = _client(state_dir=tmp_path)
    root = {"X-API-Key": "root-secret"}
    client.post("/harness/keys", json={"name": "durable"}, headers=root)
    assert (tmp_path / "keys.jsonl").exists()
    # a second app instance over the same state_dir replays the keys
    client2 = _client(state_dir=tmp_path)
    listed = client2.get("/harness/keys", headers=root).json()["data"]
    assert [k["name"] for k in listed] == ["durable"]


def test_store_rpm_window_rate_limits() -> None:
    now = [1000.0]
    store = ApiKeyStore(clock=lambda: now[0])
    raw, rec = store.mint("limited", rpm=2)
    assert rec["rpm"] == 2
    assert store.authenticate(raw) is not None
    assert store.authenticate(raw) is not None
    with pytest.raises(KeyStoreError) as excinfo:
        store.authenticate(raw)
    assert excinfo.value.code == "rate_limited"
    assert excinfo.value.retry_after == pytest.approx(60.0)
    # a refused request never counts as a use
    assert store.get(rec["key_id"])["uses"] == 2
    # the wire view never leaks the live window counters
    assert "_window_start" not in str(store.list())
    # the window rolls: past 60 s the key authenticates again
    now[0] += 61.0
    assert store.authenticate(raw)["uses"] == 3


def test_store_ttl_expires_fail_closed() -> None:
    now = [1000.0]
    store = ApiKeyStore(clock=lambda: now[0])
    raw, rec = store.mint("ephemeral", ttl_s=30.0)
    assert rec["expires_at"] == pytest.approx(1030.0)
    assert store.authenticate(raw) is not None
    now[0] += 30.0
    # dead credential — same None refusal as revoked, no oracle
    assert store.authenticate(raw) is None
    assert store.get(rec["key_id"])["uses"] == 1


def test_store_mint_rejects_bad_policy() -> None:
    store = ApiKeyStore()
    with pytest.raises(ValueError):
        store.mint(rpm=0)
    with pytest.raises(ValueError):
        store.mint(ttl_s=0.0)
    with pytest.raises(ValueError):
        store.mint(ttl_s=-5.0)


def test_store_journal_preserves_policy(tmp_path: Path) -> None:
    now = [1000.0]
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: now[0])
    _, rec = store.mint("policed", rpm=5, ttl_s=600.0)
    restored = ApiKeyStore(journal=JobJournal(path), clock=lambda: now[0])
    again = restored.get(rec["key_id"])
    assert again["rpm"] == 5 and again["expires_at"] == pytest.approx(1600.0)


def test_key_rpm_wire_429() -> None:
    client = _client()
    root = {"X-API-Key": "root-secret"}
    mint = client.post("/harness/keys", json={"rpm": 1}, headers=root)
    assert mint.status_code == 201 and mint.json()["rpm"] == 1
    raw = mint.json()["key"]
    assert client.get("/harness/commands", headers={"X-API-Key": raw}).status_code == 200
    limited = client.get("/harness/commands", headers={"X-API-Key": raw})
    assert limited.status_code == 429
    assert limited.json()["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    # the env key is not the managed key — its own path is unaffected
    assert client.get("/harness/commands", headers=root).status_code == 200


def test_key_policy_validation_wire() -> None:
    client = _client()
    root = {"X-API-Key": "root-secret"}
    assert client.post("/harness/keys", json={"rpm": 0}, headers=root).status_code == 422
    assert client.post("/harness/keys", json={"ttl_s": -1}, headers=root).status_code == 422


# ---- SDK twin --------------------------------------------------------------


def test_sdk_key_lifecycle() -> None:
    h = Fx1Harness()
    mint = h.key_create("svc")
    assert mint["key"].startswith("fx1k_") and mint["object"] == "key"
    assert h.keys()[0]["id"] == mint["id"]
    assert h.key_get(mint["id"])["name"] == "svc"
    out = h.key_revoke(mint["id"])
    assert out["enabled"] is False
    with pytest.raises(KeyError):
        h.key_get("deadbeefdeadbeef")
    with pytest.raises(KeyError):
        h.key_revoke("deadbeefdeadbeef")
    with pytest.raises(ValueError):
        h.key_revoke(mint["id"])
