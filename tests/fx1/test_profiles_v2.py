"""Credential store v2: normalized routes, precedence, and private writes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import profiles
from fx1.interactive.endpoint import EndpointSpec
from fx1.interactive.redact import DEFAULT_REGISTRY, redact_text
from fx1.serve.backends import MOONSHOT_API_URL

FX1_KEY = "sk-fx1-model-key-AAAA"
LITE_KEY = "sk-lite-model-key-BBBB"
SHARED_KEY = "sk-shared-env-key-CCCC"
MODEL_KEY = "sk-per-model-key-DDDD"
STORED_URL = "https://stored.example/v1/chat/completions"
MODEL_URL = "https://model.example/v1"
SHARED_URL = "https://shared.example/openai/v1"
COLUMNS = ("model", "host", "status", "source", "fingerprint", "set_at")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    for name in (
        profiles.ENV_API_KEY,
        profiles.ENV_BASE_URL,
        profiles.ENV_FX1_API_KEY,
        profiles.ENV_FX1_BASE_URL,
        profiles.ENV_FX1_LITE_API_KEY,
        profiles.ENV_FX1_LITE_BASE_URL,
    ):
        monkeypatch.delenv(name, raising=False)
    DEFAULT_REGISTRY.clear()
    yield tmp_path
    DEFAULT_REGISTRY.clear()


def _write_store(path: Path, payload: dict[str, Any], *, version: int | None = 1) -> None:
    body: dict[str, Any] = {"profiles": payload}
    if version is not None:
        body["schema_version"] = version
    path.write_text(json.dumps(body), encoding="utf-8")


def test_schema_v2_roundtrip_mode_and_directory(tmp_path: Path) -> None:
    target = tmp_path / "fresh" / "credentials.json"
    path, spec = profiles.set_endpoint_spec("fx1", f"  {FX1_KEY}\n", "https://API.Example.TEST/v1/")
    assert path == profiles.store_path()
    assert spec == EndpointSpec(
        "https://api.example.test",
        "/v1",
        "https://api.example.test/v1/chat/completions",
        "api.example.test",
        True,
    )
    assert spec.describe() == "POST https://api.example.test/v1/chat/completions"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    nested = profiles.set_endpoint("fx1-lite", LITE_KEY, "https://lite.example/v1", path=target)
    assert nested == target
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 2
    entry = loaded["profiles"]["fx1"]
    assert entry["api_key"] == FX1_KEY
    assert entry["base_url"] == spec.completion_url
    assert entry["service_origin"] == spec.origin
    assert entry["set_at"]
    info = profiles.resolve_endpoint_info("fx1")
    assert info is not None
    assert (info.api_key, info.completion_url) == (FX1_KEY, spec.completion_url)
    assert profiles.resolve_endpoint("fx1") == (FX1_KEY, spec.completion_url)


def test_blank_base_url_uses_default_completion_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(profiles.ENV_BASE_URL, "https://from-env.example/v1")
    profiles.set_endpoint("fx1", FX1_KEY, "   ")
    assert profiles.resolve_endpoint("fx1") == (
        FX1_KEY,
        "https://from-env.example/v1/chat/completions",
    )
    monkeypatch.delenv(profiles.ENV_BASE_URL)
    profiles.set_endpoint("fx1-lite", LITE_KEY, None)
    assert profiles.resolve_endpoint("fx1-lite") == (LITE_KEY, MOONSHOT_API_URL)


def test_validation_does_not_write_or_echo(tmp_path: Path) -> None:
    target = tmp_path / "credentials.json"
    with pytest.raises(KeyError, match="unknown model"):
        profiles.set_endpoint("fx1-nope", "x", path=target)
    with pytest.raises(ValueError, match="empty API key"):
        profiles.set_endpoint("fx1", "  \n  ", path=target)
    with pytest.raises(ValueError, match="invalid base URL") as bad_url:
        profiles.set_endpoint("fx1", FX1_KEY, "not-a-url", path=target)
    assert FX1_KEY not in str(bad_url.value)
    with pytest.raises(ValueError, match="must use HTTPS") as http_url:
        profiles.set_endpoint("fx1", FX1_KEY, "http://models.example.test/v1", path=target)
    assert FX1_KEY not in str(http_url.value)
    with pytest.raises(ValueError, match="userinfo") as userinfo:
        profiles.set_endpoint(
            "fx1",
            FX1_KEY,
            "https://user:s3cret-token@models.example.test/v1",
            path=target,
        )
    assert "s3cret-token" not in str(userinfo.value)
    assert "user:" not in str(userinfo.value)
    assert FX1_KEY not in str(userinfo.value)
    with pytest.raises(ValueError, match="whitespace or control") as junk:
        profiles.set_endpoint("fx1", "sk-bad key-value-ZZZZ", path=target)
    assert "ZZZZ" not in str(junk.value)
    assert "sk-bad" not in str(junk.value)
    assert not target.exists()


def test_failed_update_keeps_existing_bytes() -> None:
    path = profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="invalid base URL"):
        profiles.set_endpoint("fx1", LITE_KEY, "file:///invalid-endpoint")
    assert path.read_bytes() == original
    assert profiles.resolve_endpoint("fx1") == (FX1_KEY, STORED_URL)


@pytest.mark.parametrize("version", [1, None])
@pytest.mark.parametrize(
    ("entry", "completion"),
    [
        ({"key": FX1_KEY, "env_var": "MOONSHOT_API_KEY"}, MOONSHOT_API_URL),
        (
            {"api_key": FX1_KEY, "base_url": "https://Models.Example.TEST"},
            "https://models.example.test/v1/chat/completions",
        ),
        (
            {"api_key": FX1_KEY, "base_url": "https://models.example.test/v1/"},
            "https://models.example.test/v1/chat/completions",
        ),
        (
            {
                "api_key": FX1_KEY,
                "base_url": "https://models.example.test/openai/v1/chat/completions",
            },
            "https://models.example.test/openai/v1/chat/completions",
        ),
        (
            {"api_key": FX1_KEY, "base_url": "https://models.example.test/chat/completions"},
            "https://models.example.test/chat/completions",
        ),
    ],
)
def test_v1_shapes_normalize_at_read_without_rewrite(
    version: int | None, entry: dict[str, str], completion: str
) -> None:
    path = profiles.store_path()
    _write_store(path, {"fx1": entry}, version=version)
    raw = path.read_bytes()
    assert profiles.resolve_endpoint("fx1") == (FX1_KEY, completion)
    rows = {row["model"]: row for row in profiles.list_models()}
    assert rows["fx1"]["status"] == "set"
    assert rows["fx1"]["source"] == "stored"
    assert rows["fx1"]["host"] == profiles.host_of(completion)
    assert path.read_bytes() == raw


def test_remove_keeps_the_other_model_and_upgrades_schema() -> None:
    profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    profiles.set_endpoint("fx1-lite", LITE_KEY, "https://lite.example/v1")
    assert profiles.remove_endpoint("fx1") is True
    assert profiles.remove_endpoint("fx1") is False
    loaded = json.loads(profiles.store_path().read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 2
    assert "fx1" not in loaded["profiles"]
    assert profiles.resolve_endpoint("fx1-lite") == (
        LITE_KEY,
        "https://lite.example/v1/chat/completions",
    )
    missing = profiles.store_path().with_name("absent.json")
    assert profiles.remove_endpoint("fx1", path=missing) is False
    assert not missing.exists()


@pytest.mark.parametrize(
    ("stored", "env", "model", "key", "completion", "source"),
    [
        (True, {}, "fx1", FX1_KEY, STORED_URL, "stored"),
        (
            True,
            {
                profiles.ENV_FX1_API_KEY: MODEL_KEY,
                profiles.ENV_FX1_BASE_URL: MODEL_URL,
                profiles.ENV_API_KEY: SHARED_KEY,
                profiles.ENV_BASE_URL: SHARED_URL,
            },
            "fx1",
            FX1_KEY,
            STORED_URL,
            "stored",
        ),
        (
            False,
            {
                profiles.ENV_FX1_API_KEY: MODEL_KEY,
                profiles.ENV_FX1_BASE_URL: MODEL_URL,
                profiles.ENV_API_KEY: SHARED_KEY,
                profiles.ENV_BASE_URL: SHARED_URL,
            },
            "fx1",
            MODEL_KEY,
            "https://model.example/v1/chat/completions",
            "env:model",
        ),
        (
            False,
            {profiles.ENV_FX1_API_KEY: MODEL_KEY, profiles.ENV_BASE_URL: SHARED_URL},
            "fx1",
            MODEL_KEY,
            "https://shared.example/openai/v1/chat/completions",
            "env:model",
        ),
        (
            False,
            {profiles.ENV_FX1_LITE_API_KEY: MODEL_KEY},
            "fx1-lite",
            MODEL_KEY,
            MOONSHOT_API_URL,
            "env:model",
        ),
        (
            False,
            {
                profiles.ENV_FX1_BASE_URL: MODEL_URL,
                profiles.ENV_API_KEY: SHARED_KEY,
                profiles.ENV_BASE_URL: SHARED_URL,
            },
            "fx1",
            SHARED_KEY,
            "https://shared.example/openai/v1/chat/completions",
            "env:shared",
        ),
        (
            False,
            {profiles.ENV_API_KEY: SHARED_KEY, profiles.ENV_BASE_URL: SHARED_URL},
            "fx1",
            SHARED_KEY,
            "https://shared.example/openai/v1/chat/completions",
            "env:shared",
        ),
        (
            False,
            {profiles.ENV_API_KEY: SHARED_KEY},
            "fx1-lite",
            SHARED_KEY,
            MOONSHOT_API_URL,
            "env:shared",
        ),
    ],
)
def test_precedence(
    stored: bool,
    env: dict[str, str],
    model: str,
    key: str,
    completion: str,
    source: str,
) -> None:
    if stored:
        profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    info = profiles.resolve_endpoint_info(model, environ=env)
    assert info is not None
    assert info.api_key == key
    assert info.completion_url == completion
    assert info.source == source
    assert info.host == profiles.host_of(completion)
    assert profiles.resolve_endpoint(model, environ=env) == (key, completion)


def test_models_are_independent() -> None:
    env = {
        profiles.ENV_FX1_API_KEY: FX1_KEY,
        profiles.ENV_FX1_BASE_URL: "https://fx1.example/v1",
        profiles.ENV_FX1_LITE_API_KEY: LITE_KEY,
        profiles.ENV_FX1_LITE_BASE_URL: "https://lite.example/openai/v1",
    }
    fx1 = profiles.resolve_endpoint_info("fx1", environ=env)
    lite = profiles.resolve_endpoint_info("fx1-lite", environ=env)
    assert fx1 is not None and lite is not None
    assert fx1.completion_url == "https://fx1.example/v1/chat/completions"
    assert lite.completion_url == "https://lite.example/openai/v1/chat/completions"
    assert fx1.api_key == FX1_KEY
    assert lite.api_key == LITE_KEY
    assert fx1.source == lite.source == "env:model"


def test_nothing_configured_returns_none() -> None:
    assert profiles.resolve_endpoint_info("fx1", environ={}) is None
    assert profiles.resolve_endpoint("fx1", environ={profiles.ENV_FX1_BASE_URL: MODEL_URL}) is None


def test_invalid_model_url_does_not_fall_back_to_shared() -> None:
    env = {
        profiles.ENV_FX1_API_KEY: MODEL_KEY,
        profiles.ENV_FX1_BASE_URL: "http://models.example.test/v1",
        profiles.ENV_API_KEY: SHARED_KEY,
        profiles.ENV_BASE_URL: "https://shared.example/v1",
    }
    rows = {row["model"]: row for row in profiles.list_models(environ=env)}
    assert rows["fx1"]["status"] == "invalid-url"
    assert rows["fx1"]["source"] == "env:model"
    assert rows["fx1-lite"]["status"] == "env"
    assert rows["fx1-lite"]["source"] == "env:shared"
    with pytest.raises(ValueError, match="invalid DIPCATCHER_FX1_BASE_URL.*HTTPS") as exc:
        profiles.resolve_endpoint_info("fx1", environ=env)
    text = str(exc.value)
    assert MODEL_KEY not in text
    assert SHARED_KEY not in text
    assert "models.example.test" not in text


def test_invalid_stored_url_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    path = profiles.store_path()
    _write_store(
        path,
        {
            "fx1": {
                "api_key": FX1_KEY,
                "base_url": "http://user:s3cret-token@evil.example/v1",
                "set_at": "2020-01-01T00:00:00+00:00",
            }
        },
    )
    monkeypatch.setenv(profiles.ENV_API_KEY, SHARED_KEY)
    rows = {row["model"]: row for row in profiles.list_models()}
    assert rows["fx1"]["status"] == "invalid-url"
    assert rows["fx1"]["source"] == "stored"
    assert rows["fx1"]["host"] == "evil.example"
    assert rows["fx1"]["fingerprint"] == "...AAAA"
    assert rows["fx1"]["set_at"] == "2020-01-01T00:00:00+00:00"
    blob = json.dumps(rows)
    assert "s3cret-token" not in blob
    assert FX1_KEY not in blob
    with pytest.raises(ValueError, match="invalid stored base URL") as exc:
        profiles.resolve_endpoint("fx1")
    assert "HTTPS" not in str(exc.value)
    assert "userinfo" in str(exc.value)
    assert "s3cret-token" not in str(exc.value)
    assert SHARED_KEY not in str(exc.value)
    assert FX1_KEY in DEFAULT_REGISTRY.values()


def test_non_dict_stored_entry_does_not_fall_back() -> None:
    _write_store(profiles.store_path(), {"fx1": "not-an-entry"})
    rows = {
        row["model"]: row
        for row in profiles.list_models(environ={profiles.ENV_API_KEY: SHARED_KEY})
    }
    assert rows["fx1"]["status"] == "invalid-url"
    assert rows["fx1"]["source"] == "stored"
    with pytest.raises(ValueError, match="invalid stored profile"):
        profiles.resolve_endpoint_info("fx1", environ={profiles.ENV_API_KEY: SHARED_KEY})


def test_list_models_columns_and_env_fingerprint() -> None:
    profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    rows = profiles.list_models(environ={profiles.ENV_API_KEY: LITE_KEY})
    assert [tuple(row) for row in rows] == [COLUMNS, COLUMNS]
    by_model = {row["model"]: row for row in rows}
    assert by_model["fx1"]["status"] == "set"
    assert by_model["fx1"]["source"] == "stored"
    assert by_model["fx1"]["fingerprint"] == "...AAAA"
    assert by_model["fx1-lite"]["status"] == "env"
    assert by_model["fx1-lite"]["source"] == "env:shared"
    assert by_model["fx1-lite"]["fingerprint"] == "-"
    assert by_model["fx1-lite"]["set_at"] == "-"
    assert FX1_KEY not in json.dumps(rows)
    assert LITE_KEY not in json.dumps(rows)


def test_repr_str_and_environ_do_not_leak() -> None:
    env = {profiles.ENV_API_KEY: FX1_KEY, profiles.ENV_BASE_URL: MODEL_URL}
    snapshot = dict(env)
    before = {name: os.environ.get(name) for name in (profiles.ENV_API_KEY, profiles.ENV_BASE_URL)}
    info = profiles.resolve_endpoint_info("fx1", environ=env)
    assert info is not None
    assert env == snapshot
    after = {name: os.environ.get(name) for name in before}
    assert after == before
    rendered = " ".join((repr(info), str(info), format(info)))
    assert FX1_KEY not in rendered
    assert info.api_key == FX1_KEY
    assert info.fingerprint == "...AAAA"
    assert FX1_KEY not in info.fingerprint


def test_key_is_registered_for_redaction() -> None:
    DEFAULT_REGISTRY.clear()
    profiles.resolve_endpoint("fx1", environ={profiles.ENV_API_KEY: FX1_KEY})
    assert FX1_KEY in DEFAULT_REGISTRY.values()
    assert FX1_KEY not in redact_text(f"leak {FX1_KEY} end")


def test_junk_env_key_does_not_fall_back() -> None:
    env = {profiles.ENV_FX1_API_KEY: "sk-bad key-value", profiles.ENV_API_KEY: SHARED_KEY}
    with pytest.raises(ValueError, match="whitespace or control") as exc:
        profiles.resolve_endpoint_info("fx1", environ=env)
    assert "sk-bad" not in str(exc.value)
    assert SHARED_KEY not in str(exc.value)


def test_permission_self_heal() -> None:
    path = profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    os.chmod(path, 0o644)
    warnings = profiles.store_health(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert any("repaired to 0600" in warning and "0644" in warning for warning in warnings)
    os.chmod(path, 0o640)
    profiles.resolve_endpoint("fx1")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert any("0640" in warning for warning in profiles.store_health(path))


def test_permission_problem_when_chmod_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    path = profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    os.chmod(path, 0o644)

    def boom(_path: Path, _mode: int) -> None:
        raise OSError("chmod blocked")

    monkeypatch.setattr(profiles.os, "chmod", boom)
    warnings = profiles.store_health(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert any("group/world" in warning for warning in warnings)


def test_clean_store_has_no_health_warnings() -> None:
    path = profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    assert profiles.store_health(path) == []


def test_windows_acl_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text("{}\n", encoding="utf-8")
    os.chmod(path, 0o600)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0)

    assert profiles.protect_store_windows(path, runner=runner, username="ada") is False
    assert calls == []
    assert not any("Windows ACL" in warning for warning in profiles.store_health(path))

    monkeypatch.setattr(sys, "platform", "win32")
    assert profiles.protect_store_windows(path, runner=runner, username="ada") is True
    argv, kwargs = calls[0]
    assert argv == ["icacls", os.fspath(path), "/inheritance:r", "/grant:r", "ada:F"]
    assert kwargs.get("capture_output") is True
    assert kwargs.get("check") is False
    assert "shell" not in kwargs
    monkeypatch.setenv("USERNAME", "alice")
    assert profiles.protect_store_windows(path, runner=runner) is True
    assert calls[1][0][-1] == "alice:F"

    def fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, stderr=b"denied")

    assert profiles.protect_store_windows(path, runner=fail, username="ada") is False
    assert any("Windows ACL" in warning for warning in profiles.store_health(path))

    def explode(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("icacls")

    assert profiles.protect_store_windows(path, runner=explode, username="bob") is False


def test_set_endpoint_invokes_windows_protect(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def fake(path: Path, **_kwargs: Any) -> bool:
        seen.append(path)
        return True

    monkeypatch.setattr(profiles, "_is_windows", lambda: True)
    monkeypatch.setattr(profiles, "protect_store_windows", fake)
    path = profiles.set_endpoint("fx1", FX1_KEY, "https://h.example/v1")
    assert seen == [path]


def test_corrupt_store_is_quarantined_and_rewritten() -> None:
    path = profiles.store_path()
    path.write_text('{"api_key": "sk-planted-secret-value"', encoding="utf-8")
    warnings = " ".join(profiles.store_health(path))
    assert "sk-planted-secret-value" not in warnings
    assert "corrupt credentials store" in warnings
    rows = profiles.list_models()
    assert [row["status"] for row in rows] == ["unset", "unset"]
    assert not path.exists()
    quarantined = list(path.parent.glob("credentials.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "sk-planted-secret-value" in quarantined[0].read_text(encoding="utf-8")
    assert "sk-planted-secret-value" not in " ".join(profiles.store_health())
    profiles.set_endpoint("fx1", FX1_KEY, "https://fresh.example/v1")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 2
    assert loaded["profiles"]["fx1"]["api_key"] == FX1_KEY
    assert loaded["profiles"]["fx1"]["base_url"] == "https://fresh.example/v1/chat/completions"
    assert len(list(path.parent.glob("credentials.json.corrupt-*"))) == 1


def test_bad_shape_is_quarantined_on_write() -> None:
    path = profiles.store_path()
    path.write_text(json.dumps({"profiles": ["nope"]}), encoding="utf-8")
    profiles.set_endpoint("fx1", FX1_KEY, STORED_URL)
    assert profiles.resolve_endpoint("fx1") == (FX1_KEY, STORED_URL)
    assert list(path.parent.glob("credentials.json.corrupt-*"))


def test_concurrent_set_endpoint_does_not_lose_updates() -> None:
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            model = "fx1" if index % 2 == 0 else "fx1-lite"
            key = f"sk-thread-{index:02d}-secret-value"
            profiles.set_endpoint(model, key, f"https://thread{index}.example/v1")
        except BaseException as exc:  # noqa: BLE001 — the test records worker failures
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert errors == []
    loaded = json.loads(profiles.store_path().read_text(encoding="utf-8"))
    assert set(loaded["profiles"]) == {"fx1", "fx1-lite"}
    for model, entry in loaded["profiles"].items():
        assert entry["api_key"].startswith("sk-thread-")
        assert entry["base_url"].endswith("/v1/chat/completions")
        assert entry["service_origin"].startswith("https://thread")
        info = profiles.resolve_endpoint_info(model)
        assert info is not None
        assert info.api_key == entry["api_key"]
        assert info.completion_url == entry["base_url"]
        assert info.source == "stored"


def test_apply_profile_still_injects_completion_url() -> None:
    profiles.set_endpoint("fx1-lite", LITE_KEY, "https://lite.example/v1")
    target: dict[str, str] = {}
    names = profiles.apply_profile("fx1-lite", environ=target)
    assert names == [profiles.ENV_API_KEY, profiles.ENV_BASE_URL]
    assert target == {
        profiles.ENV_API_KEY: LITE_KEY,
        profiles.ENV_BASE_URL: "https://lite.example/v1/chat/completions",
    }
    with pytest.raises(RuntimeError, match="fx1"):
        profiles.apply_profile("fx1", environ={})


def test_fingerprint_delegates() -> None:
    from fx1.interactive.redact import fingerprint as redact_fingerprint

    assert profiles.fingerprint(FX1_KEY) == redact_fingerprint(FX1_KEY) == "...AAAA"
    assert FX1_KEY not in profiles.fingerprint(FX1_KEY)


def test_no_invented_artificial_hedge_host() -> None:
    profiles.set_endpoint("fx1", FX1_KEY)
    _key, url = profiles.resolve_endpoint("fx1") or ("", "")
    assert url == MOONSHOT_API_URL
    assert "artificialhedge" not in url
    assert profiles.default_base_url() == MOONSHOT_API_URL
