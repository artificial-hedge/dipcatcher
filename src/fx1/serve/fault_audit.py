"""fault_audit — adversarial-condition battery for the fx-1 harness.

Where ``api_audit`` pins the wire contract's happy paths and shapes, this
battery attacks the harness under adversarial conditions and pins what
must hold anyway:

- *Journal corruption* — malformed/truncated/reordered JSONL lines in the
  ``--state-dir`` key journal must recover-or-fail-closed *loudly*: a
  corrupt tail may drop the suffix but the store must surface that it did;
  a replayed key set must never resurrect a revoked credential and must
  never silently accept a mint it cannot persist.
- *Key-store races* — concurrent mint/revoke/authenticate/quota/token
  operations must lose no update and leave the journal byte-consistent
  with live state.
- *Lifecycle boundaries* — TTL expiry at the exact boundary and mid-call,
  ``max_requests`` at the exact budget, ``rpm`` at the window edge.
- *Wire input abuse* — oversized bodies, absurd JSON depth, NUL bytes,
  bad UTF-8, wrong content-type must produce the uniform 4xx/413 error
  envelope — never a bare 5xx.
- *Header injection* — CRLF/NUL inside ``X-API-Key`` and forged
  ``fx1k_…`` prefixes fail closed (raw-ASGI probes — httpx refuses to
  send control characters, so these drive the app scope directly).
- *BYOK smuggling* — ``base_url`` rejects non-http schemes, userinfo,
  queries, and empty netlocs; an invalid ``byok`` block is refused even
  on a non-BYOK backend.
- *Idempotency abuse* — same key + different body must 409 rather than
  return a stale cached response; a same-key concurrent submit must
  execute once, not once per racer.
- *Drain* — the latch blocks new work, ``wait_s`` is bounded, replays
  stay readable, and the latch is one-way.
- *State-dir* — a readonly or missing directory must fail loudly at boot
  (SDK twin included), never degrade silently to in-memory.

Probes are literal bools: ``True`` pins a contract that holds; ``False``
pins a measured divergence — the sealed receipt names every defect by
probe name so the finding survives byte-for-byte.

Sealed ``fault_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.harness import Harness
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore, KeyStoreError

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["fault_audit", "fault_audit_bench"]

_ROOT_KEY = "fault-audit-root"
_N = 16  # racer width — wide enough that a lost lookup window shows


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


@contextmanager
def _root_env() -> Iterator[None]:
    prev = os.environ.get("FX1_API_KEY")
    os.environ["FX1_API_KEY"] = _ROOT_KEY
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("FX1_API_KEY", None)
        else:
            os.environ["FX1_API_KEY"] = prev


def _client(
    runner: Callable[[list[str], int], tuple[int, str, str]] = _fast_runner,
) -> TestClient:
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod

    return TestClient(
        api_mod.create_app(harness=Harness(runner=runner)),
        raise_server_exceptions=False,
    )


def _run_threads(fn: Callable[[int], None], n: int = _N) -> None:
    """Run ``fn(0..n-1)`` released together behind a barrier."""
    barrier = threading.Barrier(n)

    def _w(i: int) -> None:
        barrier.wait()
        fn(i)

    ths = [threading.Thread(target=_w, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def _raw_http(
    app: Any,
    method: str,
    path: str,
    headers: list[tuple[str, bytes]],
    body: bytes = b"",
) -> dict[str, Any]:
    """Drive the ASGI app directly — httpx refuses control-char headers."""
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v) for k, v in headers],
        "client": ("127.0.0.1", 5000),
        "server": ("test", 80),
    }
    resp: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] == "http.response.start":
            resp["status"] = msg["status"]
            resp["headers"] = msg.get("headers", [])
        elif msg["type"] == "http.response.body":
            resp.setdefault("body", b"")
            resp["body"] += msg.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return resp


def _corrupt_tail(path: Path) -> None:
    """Chop the last journal line mid-record (torn-write shape)."""
    data = path.read_bytes()
    cut = data.rindex(b"\n", 0, len(data) - 1)
    path.write_bytes(data[: cut + 20])


def _corrupt_line(path: Path, idx: int) -> None:
    """Flip one byte inside line ``idx`` — invalidates its sha256."""
    lines = path.read_bytes().splitlines(keepends=True)
    line = bytearray(lines[idx])
    mark = line.find(b'"')
    line[mark + 3] = ord("X") if line[mark + 3] != ord("X") else ord("Y")
    lines[idx] = bytes(line)
    path.write_bytes(b"".join(lines))


def _swap_lines(path: Path, i: int, j: int) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    lines[i], lines[j] = lines[j], lines[i]
    path.write_bytes(b"".join(lines))


def _probe_journal_recovery() -> dict[str, bool]:
    """Corruption retains verified prefix records as disabled metadata,
    reports the break, and never restores a revoked credential."""
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "keys.jsonl"
        store = ApiKeyStore(journal=JobJournal(p))
        raw_a, rec_a = store.mint(name="a")
        _, rec_b = store.mint(name="b")
        _corrupt_tail(p)
        res = JobJournal(p).replay()
        out["journal_corrupt_tail_reports_dropped"] = res.dropped >= 1
        reloaded = ApiKeyStore(journal=JobJournal(p))
        recovered_a = reloaded.get(rec_a["key_id"])
        out["keystore_recovers_prefix_keys"] = (
            recovered_a is not None
            and recovered_a.get("enabled") is False
            and recovered_a.get("quarantined") is True
            and reloaded.authenticate(raw_a) is None
            and reloaded.get(rec_b["key_id"]) is None
        )
        # Retain verified prefix metadata, but never trust its credentials
        # after losing a record that could have carried a revocation.
        # Every dropped record must remain visible to the operator.
        out["keystore_warns_on_corrupt_journal"] = bool(getattr(reloaded, "recover_warnings", None))
        # resurrect: corrupt the revoke record itself — replay drops it
        # and everything after; the dead key must still stay dead
        p2 = Path(td) / "keys2.jsonl"
        s2 = ApiKeyStore(journal=JobJournal(p2))
        raw_v, rec_v = s2.mint(name="victim")
        s2.revoke(rec_v["key_id"])
        s2.mint(name="other")
        _corrupt_line(p2, 1)
        s3 = ApiKeyStore(journal=JobJournal(p2))
        rec_v2 = s3.get(rec_v["key_id"])
        out["revoked_key_stays_dead_through_corruption"] = (
            rec_v2 is not None
            and not rec_v2.get("enabled", True)
            and s3.authenticate(raw_v) is None
        )
        # A fresh operator-provisioned key must survive restart after
        # journal repair; quarantined historical credentials stay dead.
        raw_c, rec_c = s3.mint(name="post")
        s4 = ApiKeyStore(journal=JobJournal(p2))
        restored_c = s4.get(rec_c["key_id"])
        out["post_corruption_mint_survives_restart"] = (
            restored_c is not None
            and restored_c.get("enabled") is True
            and s4.authenticate(raw_c) is not None
            and s4.authenticate(raw_v) is None
        )
        p3 = Path(td) / "keys3.jsonl"
        s5 = ApiKeyStore(journal=JobJournal(p3))
        s5.mint(name="x")
        s5.mint(name="y")
        _swap_lines(p3, 0, 1)
        res3 = JobJournal(p3).replay()
        out["journal_reordered_lines_detected"] = res3.dropped >= 1
    return out


def _probe_store_concurrency() -> dict[str, bool]:
    """mint/revoke/authenticate/quota/token races on the in-memory store."""
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.jsonl"
        store = ApiKeyStore(journal=JobJournal(p))

        def _mint(i: int) -> None:
            store.mint(name=f"k{i}")

        _run_threads(_mint)
        res = JobJournal(p).replay()
        out["mint_race_no_lost_keys"] = (
            len(store.list()) == _N and len(res.payloads) == _N and res.dropped == 0
        )
        victims = list(store.list())[:8]

        def _rev(i: int) -> None:
            store.revoke(victims[i]["key_id"])

        _run_threads(_rev, 8)
        res2 = JobJournal(p).replay()
        replayed2 = {
            pl["record"]["key_id"]: pl["record"]["enabled"]
            for pl in res2.payloads
            if "record" in pl
        }
        live2 = {r["key_id"]: r["enabled"] for r in store.list()}
        out["mint_revoke_race_journal_consistent"] = res2.dropped == 0 and replayed2 == live2

    store = ApiKeyStore()
    raw, rec = store.mint(max_requests=4)
    hits = {"ok": 0, "refused": 0}

    def _hit(_: int) -> None:
        try:
            store.authenticate(raw)
            hits["ok"] += 1
        except KeyStoreError:
            hits["refused"] += 1

    _run_threads(_hit)
    rec_after = store.get(rec["key_id"])
    out["quota_race_no_overshoot"] = (
        hits["ok"] == 4
        and hits["refused"] == _N - 4
        and rec_after is not None
        and rec_after["uses"] == 4
    )

    _, rec2 = store.mint()

    def _charge(_: int) -> None:
        store.charge_tokens(rec2["key_id"], 3)

    _run_threads(_charge, 32)
    rec2_after = store.get(rec2["key_id"])
    out["token_charge_race_exact"] = rec2_after is not None and rec2_after["tokens_used"] == 96

    capped = ApiKeyStore(max_keys=8)
    minted: list[str] = []

    def _cap(_: int) -> None:
        with suppress(KeyStoreError):
            minted.append(capped.mint()[0])

    _run_threads(_cap)
    out["max_keys_cap_atomic_under_race"] = len(minted) == 8 and len(capped.list()) == 8
    return out


def _probe_lifecycle() -> dict[str, bool]:
    """TTL boundaries, exact quota/rpm edges, restart persistence."""
    out: dict[str, bool] = {}
    clk = {"t": 1_000.0}
    store = ApiKeyStore(clock=lambda: clk["t"])
    raw = store.mint(ttl_s=10)[0]
    clk["t"] = 1_009.9
    before = store.authenticate(raw) is not None
    clk["t"] = 1_010.0
    after = store.authenticate(raw) is None
    out["ttl_valid_before_expiry"] = before
    out["ttl_denied_at_expiry_boundary"] = after

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.jsonl"
        s1 = ApiKeyStore(journal=JobJournal(p))
        raw_q, rec_q = s1.mint(max_requests=2)
        s1.authenticate(raw_q)
        s2 = ApiKeyStore(journal=JobJournal(p))
        # A restart must restore the spent request budget from its
        # durable counter snapshot, never grant the original budget again.
        rec_q2 = s2.get(rec_q["key_id"])
        out["quota_persists_across_restart"] = rec_q2 is not None and rec_q2["uses"] == 1

    with _root_env():
        client = _client()
        r = client.post("/harness/keys", json={"max_requests": 2}, headers={"X-API-Key": _ROOT_KEY})
        key = r.json()["key"]
        codes = [
            client.get("/harness/commands", headers={"X-API-Key": key}).status_code
            for _ in range(3)
        ]
        resp = client.get("/harness/commands", headers={"X-API-Key": key})
        out["quota_exact_boundary_429_no_retry"] = (
            codes == [200, 200, 429]
            and resp.status_code == 429
            and resp.json().get("code") == "quota_exceeded"
            and resp.headers.get("Retry-After") is None
        )

        r = client.post("/harness/keys", json={"rpm": 2}, headers={"X-API-Key": _ROOT_KEY})
        key2 = r.json()["key"]
        codes2 = [
            client.get("/harness/commands", headers={"X-API-Key": key2}).status_code
            for _ in range(3)
        ]
        resp2 = client.get("/harness/commands", headers={"X-API-Key": key2})
        out["rpm_boundary_429_retry_after"] = (
            codes2 == [200, 200, 429]
            and resp2.status_code == 429
            and resp2.headers.get("Retry-After") is not None
        )
        hdrs = client.get("/harness/commands", headers={"X-API-Key": key2}).headers
        out["rpm_budget_headers_present"] = hdrs.get("X-RateLimit-Limit-Requests") == "2"

        # TTL mid-request: the credential is checked at intake; a key that
        # expires while the runner executes must not strand the in-flight call
        def sleepy(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            time.sleep(2.5)
            return 0, "ok", ""

        client2 = _client(runner=sleepy)
        r = client2.post("/harness/keys", json={"ttl_s": 2}, headers={"X-API-Key": _ROOT_KEY})
        key3 = r.json()["key"]
        run = client2.post("/harness/runs", json={"command": "doctor"}, headers={"X-API-Key": key3})
        out["ttl_midrequest_completes"] = run.status_code == 200
    return out


def _probe_wire_abuse() -> dict[str, bool]:
    """Oversized bodies, malformed JSON, NUL bytes — uniform 4xx/413."""
    out: dict[str, bool] = {}
    client = _client()
    pad = b'{"command":"doctor","pad":"' + b"A" * ((1 << 20) + 64) + b'"}'
    r = client.post("/harness/runs", content=pad, headers={"content-type": "application/json"})
    out["wire_oversized_body_413_envelope"] = (
        r.status_code == 413 and r.json().get("code") == "too_large"
    )
    r = client.post(
        "/harness/runs", content=b'{"command":', headers={"content-type": "application/json"}
    )
    out["wire_truncated_json_422"] = r.status_code == 422
    r = client.post(
        "/harness/runs",
        content=b'{"command":"doctor","x":"a\x00b"}',
        headers={"content-type": "application/json"},
    )
    out["wire_nul_string_422"] = r.status_code == 422
    r = client.post(
        "/harness/runs",
        content=b'\xff\xfe{"command":"doctor"}',
        headers={"content-type": "application/json"},
    )
    out["wire_bad_utf8_422"] = r.status_code == 422
    r = client.post(
        "/harness/runs", content=b'{"command":"doctor"}', headers={"content-type": "text/plain"}
    )
    out["wire_wrong_content_type_422"] = r.status_code == 422
    deep = b'{"command":"doctor","x":' + b"[" * 3000 + b"]" * 3000 + b"}"
    r = client.post("/harness/runs", content=deep, headers={"content-type": "application/json"})
    # divergence: json.loads recursion escapes the 4xx envelope — a bare
    # 500 instead of a structured refusal
    out["wire_deep_json_fails_closed"] = r.status_code in (400, 413, 422)

    with _root_env():
        client = _client()
        r = client.get("/harness/commands", headers={"X-API-Key": "fx1k_" + "A" * 40})
        out["wire_forged_key_prefix_401"] = r.status_code == 401
        r = client.post(
            "/harness/keys", json={"scopes": ["read"]}, headers={"X-API-Key": _ROOT_KEY}
        )
        rk = {"X-API-Key": r.json()["key"]}
        out["scope_read_denies_write_admin"] = (
            client.post("/harness/keys", json={}, headers=rk).status_code == 403
            and client.post("/harness/drain", headers=rk).status_code == 403
            and client.post("/harness/runs", json={"command": "doctor"}, headers=rk).status_code
            == 403
        )
        r = client.post("/harness/keys", json={}, headers={"X-API-Key": _ROOT_KEY})
        wk, wid = r.json()["key"], r.json()["id"]
        fresh_ok = client.get("/harness/commands", headers={"X-API-Key": wk}).status_code == 200
        self_r = client.get("/harness/self", headers={"X-API-Key": wk})
        out["self_route_identifies_key"] = (
            self_r.status_code == 200 and self_r.json().get("key", {}).get("id") == wid
        )
        client.delete(f"/harness/keys/{wid}", headers={"X-API-Key": _ROOT_KEY})
        out["revoked_key_wire_401"] = (
            fresh_ok
            and client.get("/harness/commands", headers={"X-API-Key": wk}).status_code == 401
        )
        listing = client.get("/harness/keys", headers={"X-API-Key": _ROOT_KEY}).text
        out["key_listing_never_exposes_raw"] = wk not in listing
        app = client.app
        raw_r = _raw_http(app, "GET", "/harness/commands", [("X-API-Key", b"root\r\nX: 1")])
        out["header_crlf_apikey_fails_closed"] = raw_r["status"] == 401
        raw_r = _raw_http(app, "GET", "/harness/commands", [("X-API-Key", b"root\x00x")])
        out["header_nul_apikey_fails_closed"] = raw_r["status"] == 401
    return out


def _probe_byok_smuggle() -> dict[str, bool]:
    """base_url/userinfo/query smuggling in BYOK fields — all fail closed."""
    out: dict[str, bool] = {}
    client = _client()
    bad_urls = [
        "http://user:pass@evil.example.com/v1",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://x.example.com",
        "http://",
        "https://api.example.com/v1?key=abc",
        "",
    ]
    codes = {
        client.post(
            "/harness/runs", json={"command": "doctor", "byok": {"base_url": u, "api_key": "k"}}
        ).status_code
        for u in bad_urls
    }
    out["byok_url_smuggle_refused"] = codes == {422}
    r = client.post(
        "/harness/runs",
        json={
            "command": "doctor",
            "backend": "hosted_k3",
            "byok": {"base_url": "file:///etc/passwd", "api_key": "k"},
        },
    )
    out["byok_refused_on_nonbyok_backend"] = r.status_code == 422
    return out


def _idem_race(path: str, tag: str) -> int:
    """One same-key concurrent-submit round; returns the count of
    distinct executions the server performed (1 = dedupe held)."""
    calls = {"n": 0}

    def counting(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        calls["n"] += 1
        time.sleep(0.3)
        return 0, "ok", ""

    client = _client(runner=counting)

    def _w(_: int) -> None:
        client.post(path, json={"command": "doctor"}, headers={"Idempotency-Key": tag})

    _run_threads(_w, 24)
    return calls["n"]


def _idem_race_jobs(tag: str) -> int:
    """Same-key concurrent job submits; returns distinct job ids minted."""
    client = _client()
    ids: list[str] = []
    lock = threading.Lock()

    def _w(_: int) -> None:
        r = client.post(
            "/harness/jobs", json={"command": "doctor"}, headers={"Idempotency-Key": tag}
        )
        with lock:
            ids.append(r.json().get("job_id", ""))

    _run_threads(_w, 24)
    return len(set(ids))


def _race_until_defect(run: Callable[[str], int], tag: str, rounds: int = 3) -> bool:
    """Replay the race up to ``rounds`` times; True only if dedupe held
    every single round (any double-execution is a measured defect)."""
    return all(run(f"{tag}-{i}") == 1 for i in range(rounds))


def _probe_idempotency() -> dict[str, bool]:
    """Replay/cache/conflict semantics and the concurrent-submit race."""
    out: dict[str, bool] = {}
    client = _client()
    k = {"Idempotency-Key": "fa-1"}
    r1 = client.post("/harness/runs", json={"command": "doctor"}, headers=k)
    r2 = client.post("/harness/runs", json={"command": "doctor"}, headers=k)
    out["idem_replay_returns_cached"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json().get("replayed") is True
        and r2.json().get("result") == r1.json().get("result")
    )
    r3 = client.post("/harness/runs", json={"command": "version"}, headers=k)
    out["idem_conflict_409"] = r3.status_code == 409
    out["idem_absent_reruns"] = (
        client.post("/harness/runs", json={"command": "doctor"}).status_code == 200
    )

    client.post("/harness/drain")
    r4 = client.post("/harness/runs", json={"command": "doctor"}, headers=k)
    out["idem_replay_survives_drain"] = r4.status_code == 200 and r4.json().get("replayed") is True
    r5 = client.post("/harness/runs", json={"command": "version"}, headers=k)
    out["idem_conflict_under_drain"] = r5.status_code == 409

    # same-key concurrent submit: every racer must see ONE execution —
    # measured: the lookup→put window lets threads miss the cache
    out["idem_race_single_execution"] = _race_until_defect(
        lambda tag: _idem_race("/harness/runs", tag), "fa-race"
    )
    out["idem_race_jobs_single_id"] = _race_until_defect(_idem_race_jobs, "fa-job")
    return out


def _probe_drain() -> dict[str, bool]:
    """Drain latch: blocks new work, bounded wait, reads stay open."""
    out: dict[str, bool] = {}

    def sleepy(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        time.sleep(0.4)
        return 0, "ok", ""

    client = _client(runner=sleepy)
    jr = client.post("/harness/jobs", json={"command": "doctor"})
    jid = jr.json()["job_id"]
    t0 = time.monotonic()
    dr = client.post("/harness/drain", params={"wait_s": 0.1})
    elapsed = time.monotonic() - t0
    out["drain_wait_bounded"] = (
        dr.status_code == 200
        and dr.json()["drained"] is False
        and dr.json()["inflight"] >= 1
        and elapsed < 2.0
    )
    out["drain_blocks_new_submit"] = (
        client.post("/harness/runs", json={"command": "doctor"}).status_code == 503
        and client.post("/harness/jobs", json={"command": "doctor"}).json().get("code")
        == "draining"
    )
    out["drain_keeps_job_reads_open"] = client.get(f"/harness/jobs/{jid}").status_code == 200
    out["ready_fails_under_drain"] = client.get("/ready").status_code == 503
    dr2 = client.post("/harness/drain", params={"wait_s": 1.0})
    out["drain_one_way_latch"] = (
        dr2.json()["drained"] is True
        and client.post("/harness/runs", json={"command": "doctor"}).status_code == 503
    )
    return out


def _probe_state_dir() -> dict[str, bool]:
    """readonly/missing state dirs must fail loudly — never go in-memory."""
    out: dict[str, bool] = {}
    import fx1.serve.api as api_mod
    from fx1.sdk import Fx1Harness

    with tempfile.TemporaryDirectory() as td:
        ro = Path(td) / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        try:
            try:
                api_mod.create_app(harness=Harness(runner=_fast_runner), state_dir=ro)
                out["state_dir_readonly_fails_loud"] = False
            except OSError:
                out["state_dir_readonly_fails_loud"] = True
            try:
                Fx1Harness(state_dir=ro)
                out["sdk_readonly_state_dir_fails_loud"] = False
            except OSError:
                out["sdk_readonly_state_dir_fails_loud"] = True
        finally:
            ro.chmod(0o755)

        missing = Path(td) / "no" / "such" / "dir"
        try:
            api_mod.create_app(harness=Harness(runner=_fast_runner), state_dir=missing)
            out["state_dir_missing_created"] = missing.exists()
        except OSError:
            out["state_dir_missing_created"] = False

        sd = Path(td) / "live"
        sd.mkdir()
        with _root_env():
            app = api_mod.create_app(harness=Harness(runner=_fast_runner), state_dir=sd)
            from fastapi.testclient import TestClient

            client = TestClient(app, raise_server_exceptions=False)
            sd.chmod(0o555)
            try:
                r = client.post("/harness/keys", json={}, headers={"X-API-Key": _ROOT_KEY})
                # a mint over a now-readonly journal must not report
                # success while the record never persisted
                out["mint_after_state_dir_readonly_fails_loud"] = r.status_code >= 500
            finally:
                sd.chmod(0o755)
    return out


def fault_audit() -> dict[str, Any]:
    """Run every probe; results are literal bools keyed by probe name."""
    prev = os.environ.pop("FX1_API_KEY", None)
    out: dict[str, Any] = {}
    try:
        for section in (
            _probe_journal_recovery,
            _probe_store_concurrency,
            _probe_lifecycle,
            _probe_wire_abuse,
            _probe_byok_smuggle,
            _probe_idempotency,
            _probe_drain,
            _probe_state_dir,
        ):
            out.update(section())
    finally:
        if prev is not None:
            os.environ["FX1_API_KEY"] = prev
    return out


def fault_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = fault_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True)
    out: dict[str, Any] = {
        "kind": "fault_audit",
        "schema": "fault_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Harness holds under adversarial conditions: journal "
            "corruption retains verified prefix metadata, quarantines "
            "old credentials and reports it; fresh replacement keys and "
            "quota counters survive restart, key races lose no updates, "
            "TTL/quota/rpm boundaries are exact, "
            "wire abuse gets the uniform error envelope, header and BYOK "
            "smuggling fail closed, idempotent replays and conflicts "
            "survive drain, and unreadable state dirs fail loudly."
            if ok
            else f"FAULT AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(fault_audit_bench(), indent=2, sort_keys=True))
