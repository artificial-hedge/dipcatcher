"""Deploy-gate honesty audit for ``fx1 harness selftest``/``bench`` — lane 171.

CLAIM UNDER TEST — the selftest IS the deploy gate: a user running
``fx1 harness selftest`` trusts the verdict, so the gate itself must be
honest. This battery drives the real surfaces end to end and asserts:

- *exit semantics* — a clean deployment exits 0 with an all-True
  report; deliberately broken targets (dead URL, a 500-storm, a wrong
  key, an app whose auth gate is off) fail with named checks, never a
  blanket pass. An empty report is not ok.
- *check isolation* — one failing check can't mask the rest; a check
  that crashes reports the exception class as a failure, never
  swallowed; a falsy result fails honestly with the ``got`` value.
- *remote read-only* — ``--remote`` sends GETs plus advisory POSTs
  (``/harness/score``, ``/harness/gate/check``); the one mutating verb
  attempted — a bare no-key ``complete`` — is refused 401 by the gate.
  A live deployment's journals are byte-identical after the run.
- *env/state isolation* — a local run scrubs BYOK + MOONSHOT env and
  restores it after (a real defect found here: ``MOONSHOT_API_KEY``
  was popped and never restored); ``--state-dir`` confines every write
  to the dir.
- *determinism/reentrancy* — repeated runs agree; concurrent local
  runs can't corrupt each other's env (``_LOCAL_RUN_LOCK`` — a second
  defect: parallel runs raced on shared ``os.environ``).
- *selftest↔gate parity* — the auth gate, error envelope, version
  negotiation, and receipt-verify assertions are the same calls a real
  client makes, measured against a live app on loopback.
- *bench honesty* — the latency card is measured values, the error
  histogram classifies by exception class, a refusal storm is a
  measured error (exit 1 — never a pass), prompt text never leaves the
  record (sha256 only), and parameter bounds refuse before a single
  request flies.
- *doctor flags* — presence flags only: a secret is ``set``/``unset``,
  never its value; ``fx1 harness doctor`` does not exist.

Sealed ``selftest_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fx1.selftest import SelftestReport, _run, _wait_job, run_selftest
from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _StubBackend,
    _temporary_directory,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["selftest_audit", "selftest_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_BYOK_URL_ENV = "FX1_BYOK_BASE_URL"
_BYOK_KEY_ENV = "FX1_BYOK_API_KEY"
_BYOK_MODEL_ENV = "FX1_BYOK_MODEL"
_MOON_ENV = "MOONSHOT_API_KEY"
_SELFTEST_ENVS = (
    _API_KEY_ENV,
    _BYOK_URL_ENV,
    _BYOK_KEY_ENV,
    _BYOK_MODEL_ENV,
    _MOON_ENV,
)
_KEY = "selftest-audit-key"  # a probe string, not a credential
_WRONG_KEY = "selftest-wrong-key"
_MOON_SENTINEL = "moonshot-sentinel-value"
_BYOK_SENTINEL = "byok-model-sentinel"
_DEAD = "http://127.0.0.1:1"  # NOSONAR — an intentional dead target
_DOCTOR = "doctor"
_AUTH_CHECK = "auth_gate_rejects_missing_key"
_LOCAL_CHECKS = frozenset(
    {
        "boot",
        "health_byok",
        "version_matches_wire",
        "commands",
        _AUTH_CHECK,
        "error_envelope_maps_valueerror",
        "run_command",
        "complete_byok",
        "stream_sse_reassembles",
        "idempotent_replay",
        "job_lifecycle",
        "job_receipt_verifies",
        "advisory_surfaces",
        "inprocess_parity",
        "drain_latches",
        "drained_rejects_work",
    }
)
_REMOTE_CHECKS = frozenset(
    {"health", "version_matches_wire", "commands", "score_advisory", "gate_preflight"}
)
#: Routes a remote selftest may POST: the advisory checks, plus the bare
#: no-key ``complete`` whose 401 IS the auth-gate proof.
_ADVISORY_POSTS = frozenset({"/harness/score", "/harness/gate/check"})
_REFUSED_POST = "/harness/complete"
#: Write families that must never appear on the remote wire at all.
_MUTATING_PREFIXES = (
    "/harness/jobs",
    "/harness/runs",
    "/harness/keys",
    "/harness/drain",
    "/harness/files",
    "/harness/uploads",
    "/v1/",
)
#: Generous ceiling — the dead-target probes catch a hang (connect loops,
#: missing timeouts), not a slow CI box.
_DEAD_RUN_BUDGET_S = 30.0


def _resources() -> ExitStack:
    return _RESOURCES.get()


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=15)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` as ``relpath -> bytes`` — a byte-exact
    snapshot so remote-mode probes can prove zero state mutation."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class _RequestLog:
    """ASGI tap recording ``(method, path, status)`` per request — the
    wire-level evidence that remote probes are read-only."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.calls: list[tuple[str, str, int]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        status = 0

        async def _send(message: Any) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 0))
            await send(message)

        try:
            await self._app(scope, receive, _send)
        finally:
            self.calls.append((method, path, status))


class _Blackhole(BaseHTTPRequestHandler):
    """Every request answers 500 — an 'up' deployment that is broken."""

    protocol_version = "HTTP/1.1"
    server_version = "blackhole"

    def _fail(self) -> None:
        payload = b'{"error":"broken"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 — stdlib hook name
        self._fail()

    def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
        self._fail()

    def log_message(self, *args: Any) -> None:  # keep output quiet
        return None


def _serve_handler(handler: type[BaseHTTPRequestHandler]) -> str:
    """A raw loopback HTTP server; returns ``http://127.0.0.1:<port>``."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    res = _resources()
    res.callback(srv.server_close)  # LIFO: runs last — after the loop stops
    res.callback(thread.join, 15)
    res.callback(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}"


def _live_app(*, keyed: bool = True) -> tuple[str, _RequestLog, Path]:
    """A real deployment on loopback: the production app + middleware
    stack with a stub backend; env key when ``keyed``. Returns
    ``(base_url, request_log, state_dir)``."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.e2e_audit import _serve_uvicorn  # noqa: PLC0415

    resources = _resources()
    state_dir = _temporary_directory() / "state"
    saved = os.environ.get(_API_KEY_ENV)
    try:
        if keyed:
            os.environ[_API_KEY_ENV] = _KEY
        else:
            os.environ.pop(_API_KEY_ENV, None)

        def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            del timeout_s
            return 0, "ran:" + " ".join(argv), ""

        app = api_mod.create_app(
            harness=Harness(runner=_runner),
            backend_resolver=lambda name, *a, **k: _StubBackend("stub-v0"),
            state_dir=state_dir,
            receipts_dir=_temporary_directory() / "receipts",
            ft_dir=_temporary_directory() / "ft",
        )
        spy = _RequestLog(app)
        server, thread, port = _serve_uvicorn(spy)
        resources.callback(_stop_server, server, thread)
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return f"http://127.0.0.1:{port}", spy, state_dir
    finally:
        if saved is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved


# ---------------------------------------------------------------------------
# Report contract — the verdict object's own semantics
# ---------------------------------------------------------------------------


def _report_contract() -> dict[str, bool]:
    out: dict[str, bool] = {}
    empty = SelftestReport(mode="local")
    out["empty_report_not_ok"] = empty.ok is False
    partial = SelftestReport(mode="local")
    partial.note("a", True)
    partial.note("b", False, "boom")
    out["one_failure_flips_ok"] = partial.ok is False
    out["note_records_verdict"] = [c.name for c in partial.checks] == [
        "a",
        "b",
    ] and partial.checks[1].detail == "boom"
    d = partial.as_dict()
    out["as_dict_json_roundtrip"] = json.loads(json.dumps(d)) == d
    out["as_dict_lists_every_check"] = len(d["checks"]) == 2 and d["ok"] is False

    def _boom() -> None:
        raise RuntimeError("check exploded")

    checks: list[Any] = []
    _run(checks, "crashy", _boom)
    _run(checks, "after_crash", lambda: True)
    _run(checks, "falsy", lambda: 0)
    _run(checks, "mismatch", lambda: 1, 2)
    out["crashed_check_is_named_failure"] = (
        checks[0].ok is False and "RuntimeError" in checks[0].detail
    )
    out["crash_does_not_mask_later_checks"] = checks[1].ok is True
    out["falsy_result_fails_with_got"] = checks[2].ok is False and "got 0" in checks[2].detail
    out["expect_mismatch_reports_got"] = checks[3].ok is False and "got 1" in checks[3].detail
    return out


# ---------------------------------------------------------------------------
# Local golden path — a clean deployment passes with the pinned check set
# ---------------------------------------------------------------------------


def _local_run_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    report = run_selftest()
    names = [c.name for c in report.checks]
    out["local_report_ok"] = report.ok is True
    out["local_mode_labeled"] = report.mode == "local"
    out["local_check_set"] = set(names) == _LOCAL_CHECKS
    out["local_checks_unique_named"] = len(names) == len(set(names)) and all(names)
    out["local_failures_named"] = [c.name for c in report.checks if not c.ok] == []
    d = report.as_dict()
    out["local_as_dict_honest"] = (
        d["ok"] is True
        and len(d["checks"]) == len(report.checks)
        and json.loads(json.dumps(d)) == d
    )
    rep2 = run_selftest()
    out["local_deterministic"] = [(c.name, c.ok) for c in rep2.checks] == [
        (c.name, c.ok) for c in report.checks
    ]
    return out


# ---------------------------------------------------------------------------
# Env isolation — the run scrubs ambient config and restores it after
# ---------------------------------------------------------------------------


def _env_isolation_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    os.environ[_MOON_ENV] = _MOON_SENTINEL
    os.environ[_BYOK_MODEL_ENV] = _BYOK_SENTINEL
    os.environ[_API_KEY_ENV] = "ambient-" + _KEY
    try:
        report = run_selftest()
        out["env_run_still_ok"] = report.ok is True
        # DEFECT pin: run_selftest popped MOONSHOT_API_KEY and never
        # restored it — the caller's hosted-eval key vanished.
        out["moonshot_key_restored"] = os.environ.get(_MOON_ENV) == _MOON_SENTINEL
        out["byok_env_restored"] = os.environ.get(_BYOK_MODEL_ENV) == _BYOK_SENTINEL
        out["api_key_env_restored"] = os.environ.get(_API_KEY_ENV) == "ambient-" + _KEY
    finally:
        for name in (_MOON_ENV, _BYOK_MODEL_ENV, _API_KEY_ENV):
            os.environ.pop(name, None)

    # A dead ambient BYOK URL can't be what answered: the run passes only
    # because it binds its own loopback stub — it never hits an upstream.
    os.environ[_BYOK_URL_ENV] = _DEAD + "/v1"
    os.environ[_BYOK_KEY_ENV] = "ambient-dead-key"
    try:
        report2 = run_selftest()
        out["byok_binds_own_stub"] = report2.ok is True
        out["byok_ambient_env_restored"] = (
            os.environ.get(_BYOK_URL_ENV) == _DEAD + "/v1"
            and os.environ.get(_BYOK_KEY_ENV) == "ambient-dead-key"
        )
    finally:
        os.environ.pop(_BYOK_URL_ENV, None)
        os.environ.pop(_BYOK_KEY_ENV, None)
    return out


# ---------------------------------------------------------------------------
# Reentrancy — concurrent local runs can't corrupt shared env
# ---------------------------------------------------------------------------


def _concurrency_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    # DEFECT pin: two parallel local runs raced on os.environ — run A's
    # env restore mid-run blanked BYOK config under run B's checks.
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(lambda _i: run_selftest(), range(2))
    out["concurrent_runs_both_ok"] = a.ok is True and b.ok is True
    out["concurrent_same_check_vector"] = [(c.name, c.ok) for c in a.checks] == [
        (c.name, c.ok) for c in b.checks
    ]
    out["concurrent_env_clean_after"] = all(os.environ.get(name) is None for name in _SELFTEST_ENVS)
    return out


# ---------------------------------------------------------------------------
# State isolation — --state-dir confines every write
# ---------------------------------------------------------------------------


def _state_dir_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = _temporary_directory()
    state_dir = root / "state"
    report = run_selftest(state_dir=str(state_dir))
    names = [c.name for c in report.checks]
    out["state_dir_run_ok"] = report.ok is True
    out["state_dir_adds_restart_check"] = "restart_recovers_job" in names
    out["state_dir_check_count"] = len(names) == len(_LOCAL_CHECKS) + 1
    stray = [p for p in root.rglob("*") if p.is_file() and not p.is_relative_to(state_dir)]
    out["state_writes_confined"] = stray == []
    nonempty = {p.name for p in state_dir.rglob("*") if p.is_file() and p.stat().st_size > 0}
    out["state_nonempty_journals"] = nonempty <= {"jobs.jsonl", "idem_runs.jsonl"}
    out["state_job_journal_written"] = "jobs.jsonl" in nonempty
    # the gate writes no verdict artifact — the report lives in-process
    out["selftest_writes_no_report_artifact"] = not any(
        p.suffix == ".json" for p in root.rglob("*") if p.is_file()
    )
    return out


# ---------------------------------------------------------------------------
# Remote mode — read-only probes over a real loopback socket
# ---------------------------------------------------------------------------


def _remote_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    base, spy, state_dir = _live_app()
    journals_before = _tree_bytes(state_dir)

    report = run_selftest(remote=base, api_key=_KEY)
    names = {c.name for c in report.checks}
    out["remote_mode_labeled"] = report.mode == "remote"
    out["remote_clean_ok"] = report.ok is True
    out["remote_check_set"] = names == _REMOTE_CHECKS | {_AUTH_CHECK}

    posts = {(m, p, s) for m, p, s in spy.calls if m == "POST"}
    out["remote_methods_readonly"] = all(m in {"GET", "POST"} for m, _, _ in spy.calls)
    out["remote_posts_advisory_or_refused"] = all(
        p in _ADVISORY_POSTS or (p == _REFUSED_POST and s == 401) for _, p, s in posts
    )
    out["remote_no_mutating_route_2xx"] = not any(
        any(p.startswith(pre) for pre in _MUTATING_PREFIXES) and s < 400 for _, p, s in spy.calls
    )
    out["remote_state_untouched"] = _tree_bytes(state_dir) == journals_before

    rep2 = run_selftest(remote=base, api_key=_KEY)
    out["remote_deterministic"] = [(c.name, c.ok) for c in rep2.checks] == [
        (c.name, c.ok) for c in report.checks
    ]

    bare = run_selftest(remote=base)  # no key on a keyed deployment
    out["remote_no_key_cannot_pass"] = bare.ok is False
    out["remote_no_key_names_refusals"] = {c.name for c in bare.checks if not c.ok} == (
        _REMOTE_CHECKS - {"health"}
    )
    out["remote_no_key_health_still_public"] = any(c.name == "health" and c.ok for c in bare.checks)
    out["remote_no_key_skips_auth_check"] = _AUTH_CHECK not in {c.name for c in bare.checks}

    wrong = run_selftest(remote=base, api_key=_WRONG_KEY)
    out["remote_wrong_key_fails"] = wrong.ok is False
    # the auth-gate check itself still passes — the gate IS refusing
    out["remote_wrong_key_gate_still_honest"] = any(
        c.name == _AUTH_CHECK and c.ok for c in wrong.checks
    )

    # An unauth-gated deployment: advisory checks pass but the auth
    # proof must fail — the gate can't be skipped silently.
    base2, _spy2, _sd2 = _live_app(keyed=False)
    unauthed = run_selftest(remote=base2, api_key=_KEY)
    auth_checks = [c for c in unauthed.checks if c.name == _AUTH_CHECK]
    out["remote_unauthed_caught"] = (
        unauthed.ok is False and len(auth_checks) == 1 and auth_checks[0].ok is False
    )
    out["remote_unauthed_advisories_pass"] = all(
        c.ok for c in unauthed.checks if c.name != _AUTH_CHECK
    )
    open_run = run_selftest(remote=base2)
    out["remote_open_deployment_advisories_ok"] = open_run.ok is True
    out["remote_open_auth_unproven"] = _AUTH_CHECK not in {c.name for c in open_run.checks}

    # A dead target is honest failure, never a hang.
    t0 = time.monotonic()
    dead = run_selftest(remote=_DEAD, timeout_s=1.0)
    dead_s = time.monotonic() - t0
    out["remote_dead_not_ok"] = dead.ok is False
    out["remote_dead_failures_named"] = len(dead.checks) >= len(_REMOTE_CHECKS) and all(
        c.name != "" and not c.ok for c in dead.checks
    )
    out["remote_dead_returns_bounded"] = dead_s < _DEAD_RUN_BUDGET_S

    # An up-but-broken deployment (500 on everything) fails named checks.
    storm = run_selftest(remote=_serve_handler(_Blackhole), timeout_s=2.0)
    out["remote_500s_not_ok"] = storm.ok is False
    out["remote_500s_failures_named"] = bool(storm.checks) and all(
        c.name != "" and not c.ok for c in storm.checks
    )
    return out


# ---------------------------------------------------------------------------
# Selftest ↔ gate parity — the checks are the client's real calls
# ---------------------------------------------------------------------------


def _gate_parity_probes() -> dict[str, bool]:
    from fx1.serve.client import (  # noqa: PLC0415
        HarnessAuthError,
        HarnessClient,
    )
    from fx1.serve.contract import API_VERSION  # noqa: PLC0415
    from fx1.serve.e2e_audit import _raises  # noqa: PLC0415

    out: dict[str, bool] = {}
    base, _spy, _sd = _live_app()
    client = HarnessClient(base, api_key=_KEY, timeout_s=10.0)
    bare = HarnessClient(base, timeout_s=10.0)
    msg = [{"role": "user", "content": "selftest"}]
    out["parity_version_field"] = client.server_version().get("api_version") == API_VERSION
    out["parity_missing_key_is_auth_error"] = (
        _raises(lambda: bare.complete(msg, backend="byok")) == HarnessAuthError.__name__
    )
    out["parity_wrong_key_is_auth_error"] = (
        _raises(
            lambda: HarnessClient(base, api_key=_WRONG_KEY, timeout_s=10.0).complete(
                msg, backend="byok"
            )
        )
        == HarnessAuthError.__name__
    )
    out["parity_bogus_backend_maps_valueerror"] = (
        _raises(lambda: client.complete(msg, backend="bogus")) == "ValueError"
    )
    out["parity_complete_byok_works"] = client.complete(msg, backend="byok").content.startswith(
        "stub:"
    )
    job_id = client.submit_run(_DOCTOR)
    final = _wait_job(client, job_id, 10.0)
    out["parity_receipt_verifies"] = (
        final.get("status") == "succeeded"
        and client.verify_receipt(client.job_receipt(job_id)).valid
    )
    out["parity_advisory_surfaces"] = (
        client.score("the sharpe ratio was 9.9")[0].get("total", 0) < 0
        and client.check_text("the sharpe ratio was 9.9").ok is False
    )
    return out


# ---------------------------------------------------------------------------
# Bench honesty — measured errors, real latency card, honest histogram
# ---------------------------------------------------------------------------


class _EchoResult:
    """Synthetic completion result — carries usage/model/backend like
    the real surface's, so token accounting is measurable."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage = {"prompt_tokens": 5, "completion_tokens": 3}
        self.model = "stub-v0"
        self.backend = "byok"


class _EchoSurface:
    """Synthetic ``_Completer`` — counts calls, echoes the last turn."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str,
        max_tokens: int | None,
        timeout_s: float | None,
        seed: int | None,
        byok: dict[str, str] | None,
    ) -> _EchoResult:
        del backend, max_tokens, timeout_s, seed, byok
        self.calls += 1
        return _EchoResult("stub:" + messages[-1]["content"])


class _RefuseSurface(_EchoSurface):
    """Every call raises ``BackendNotConfiguredError`` — a refusal storm."""

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> _EchoResult:
        from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

        del messages, kw
        raise BackendNotConfiguredError("drained")


class _MixedSurface(_EchoSurface):
    """Alternating failure classes — the histogram's buckets stay distinct."""

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> _EchoResult:
        del messages, kw
        self.calls += 1
        if self.calls % 3 == 0:
            raise KeyError("gone")
        if self.calls % 2 == 0:
            raise ValueError("bad")
        return _EchoResult("ok")


def _bench_unit_probes() -> dict[str, bool]:
    from fx1.harness_bench import run_bench  # noqa: PLC0415
    from fx1.serve.e2e_audit import _raises  # noqa: PLC0415

    out: dict[str, bool] = {}
    echo = _EchoSurface()

    clean = run_bench(echo, n=4, concurrency=1, warmup=1, prompt="ping", seed=7)
    m = clean["metrics"]
    out["bench_clean_zero_errors"] = m["error_count"] == 0 and m["errors"] == {}
    out["bench_measured_requests_exact"] = m["measured_requests"] == 4
    out["bench_warmup_unmeasured"] = echo.calls == 5
    lat = m["latency_s"]
    out["bench_latency_card_ordered"] = (
        0.0 <= lat["min"] <= lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"] <= lat["max"]
    )
    out["bench_latency_wall_measured"] = lat["max"] > 0.0 and m["wall_s"] > 0.0
    out["bench_prompt_hashed_not_embedded"] = clean["params"]["prompt_sha256"] == hashlib.sha256(
        b"ping"
    ).hexdigest() and "ping" not in json.dumps(clean["params"])
    out["bench_mode_recorded"] = clean["mode"] == "in_process"
    out["bench_tokens_accounted"] = (
        m["prompt_tokens_total"] == 20
        and m["completion_tokens_total"] == 12
        and m["usage_reported"] == 4
    )
    out["bench_models_backends_reported"] = m["models"] == ["stub-v0"] and m["backends"] == ["byok"]

    # A refusal storm is a measured error — every class named, the run
    # bounded, never a pass.
    t0 = time.monotonic()
    storm = run_bench(_RefuseSurface(), n=3, concurrency=2, warmup=0, prompt="ping")
    storm_s = time.monotonic() - t0
    sm = storm["metrics"]
    out["bench_refusal_storm_measured_error"] = sm["error_count"] == 3
    out["bench_refusal_rate_honest"] = math.isclose(sm["error_rate"], 1.0)
    out["bench_refusal_class_honest"] = sm["errors"] == {"BackendNotConfiguredError": 3}
    out["bench_refusal_returns_bounded"] = storm_s < _DEAD_RUN_BUDGET_S

    # Mixed classes land in distinct histogram buckets — never conflated.
    mixed = run_bench(_MixedSurface(), n=6, concurrency=1, warmup=0, prompt="ping")
    out["bench_histogram_per_class"] = mixed["metrics"]["errors"] == {
        "KeyError": 2,
        "ValueError": 2,
    }
    out["bench_mixed_ok_counted_honestly"] = (
        mixed["metrics"]["measured_requests"] == 6 and mixed["metrics"]["error_count"] == 4
    )

    # Bounds refuse before a single request spends.
    bounded = _EchoSurface()
    for probe, kw in (
        ("bench_bounds_n_low", {"n": 0}),
        ("bench_bounds_n_high", {"n": 4097}),
        ("bench_bounds_concurrency", {"concurrency": 0}),
        ("bench_bounds_warmup", {"warmup": 257}),
        ("bench_bounds_max_tokens", {"max_tokens": 0}),
        ("bench_bounds_timeout", {"timeout_s": 0.0}),
        ("bench_bounds_prompt_empty", {"prompt": "   "}),
        ("bench_bounds_prompt_long", {"prompt": "x" * 32769}),
        ("bench_bounds_mode", {"mode": "bogus"}),
    ):
        args: dict[str, Any] = {"prompt": "ping", **kw}
        out[probe] = _raises(lambda args=args: run_bench(bounded, **args)) == "ValueError"
    out["bench_bounds_check_before_spend"] = bounded.calls == 0
    return out


# ---------------------------------------------------------------------------
# CLI exit-code honesty — the deploy-gate surface users actually invoke
# ---------------------------------------------------------------------------


def _cli_probes() -> dict[str, bool]:
    from typer.testing import CliRunner  # noqa: PLC0415

    from fx1.cli import app as cli_app  # noqa: PLC0415
    from fx1.serve.client import HarnessClient  # noqa: PLC0415

    out: dict[str, bool] = {}
    runner = CliRunner()
    base, _spy, _sd = _live_app()

    clean = runner.invoke(cli_app, ["harness", "selftest"])
    out["cli_local_exit0"] = clean.exit_code == 0
    payload = json.loads(clean.output)
    out["cli_report_ok_field"] = payload["ok"] is True and payload["mode"] == "local"
    out["cli_report_lists_checks"] = len(payload["checks"]) == len(_LOCAL_CHECKS)

    dead = runner.invoke(cli_app, ["harness", "selftest", "--remote", _DEAD, "--timeout", "1"])
    out["cli_dead_remote_exit2"] = dead.exit_code == 2
    dpayload = json.loads(dead.output)
    out["cli_dead_report_names_failures"] = dpayload["ok"] is False and any(
        not c["ok"] for c in dpayload["checks"]
    )

    live = runner.invoke(cli_app, ["harness", "selftest", "--remote", base, "--api-key", _KEY])
    out["cli_live_remote_exit0"] = live.exit_code == 0 and json.loads(live.output)["ok"] is True
    wrong = runner.invoke(
        cli_app, ["harness", "selftest", "--remote", base, "--api-key", _WRONG_KEY]
    )
    out["cli_wrong_key_exit2"] = wrong.exit_code == 2

    bench_args = [
        "harness",
        "bench",
        "--remote",
        base,
        "--api-key",
        _KEY,
        "--backend",
        "byok",
        "--n",
        "3",
        "--warmup",
        "0",
    ]
    ok = runner.invoke(cli_app, bench_args)
    out["cli_bench_live_exit0"] = ok.exit_code == 0
    rec = json.loads(ok.output)
    out["cli_bench_live_measured"] = (
        rec["metrics"]["measured_requests"] == 3
        and rec["metrics"]["error_count"] == 0
        and rec["mode"] == "remote"
    )
    deadb = runner.invoke(
        cli_app,
        ["harness", "bench", "--remote", _DEAD, "--n", "2", "--warmup", "0", "--timeout", "1"],
    )
    out["cli_bench_dead_exit1"] = deadb.exit_code == 1
    out["cli_bench_dead_histogram_honest"] = (
        "HarnessTransportError" in json.loads(deadb.output)["metrics"]["errors"]
    )
    wrongb = runner.invoke(
        cli_app,
        [
            "harness",
            "bench",
            "--remote",
            base,
            "--api-key",
            _WRONG_KEY,
            "--n",
            "2",
            "--warmup",
            "0",
        ],
    )
    out["cli_bench_wrong_key_exit1"] = wrongb.exit_code == 1
    out["cli_bench_wrong_key_histogram"] = (
        "HarnessAuthError" in json.loads(wrongb.output)["metrics"]["errors"]
    )

    # A drained deployment is a refusal storm the bench can't pass on.
    base3, _spy3, _sd3 = _live_app()
    HarnessClient(base3, api_key=_KEY, timeout_s=10.0).drain()
    drained = runner.invoke(
        cli_app,
        ["harness", "bench", "--remote", base3, "--api-key", _KEY, "--n", "2", "--warmup", "0"],
    )
    out["cli_bench_drained_exit1"] = drained.exit_code == 1
    out["cli_bench_drained_histogram"] = (
        "BackendNotConfiguredError" in json.loads(drained.output)["metrics"]["errors"]
    )

    # --receipt prints the sealed fx1_bench_result.v1 doc, verifiable.
    sealed = runner.invoke(cli_app, [*bench_args, "--receipt"])
    doc = json.loads(sealed.output)
    receipt_path = _temporary_directory() / "bench_receipt.json"
    receipt_path.write_text(json.dumps(doc))
    from quant_fund.research.receipt_v2 import verify_receipt_file  # noqa: PLC0415

    out["cli_bench_receipt_verifies"] = (
        sealed.exit_code == 0
        and doc.get("schema") == "fx1_bench_result.v1"
        and verify_receipt_file(receipt_path)["valid"] is True
    )
    return out


# ---------------------------------------------------------------------------
# Doctor — presence flags, never values
# ---------------------------------------------------------------------------


def _doctor_probes() -> dict[str, bool]:
    from typer.testing import CliRunner  # noqa: PLC0415

    from fx1.cli import app as cli_app  # noqa: PLC0415

    out: dict[str, bool] = {}
    runner = CliRunner()
    res = runner.invoke(cli_app, [_DOCTOR])
    out["doctor_exit0"] = res.exit_code == 0
    payload = json.loads(res.output)
    out["doctor_flags_honest"] = payload.get("moonshot_key") == "unset"
    out["doctor_json_serializable"] = isinstance(payload, dict) and bool(payload)

    os.environ[_MOON_ENV] = _MOON_SENTINEL
    try:
        flagged = runner.invoke(cli_app, [_DOCTOR])
        out["doctor_set_flag_honest"] = json.loads(flagged.output)["moonshot_key"] == "set"
        # presence only — the secret's VALUE never appears in output
        out["doctor_never_leaks_value"] = _MOON_SENTINEL not in flagged.output
    finally:
        os.environ.pop(_MOON_ENV, None)

    # `fx1 harness doctor` does not exist — the gate doesn't pretend a
    # deeper diagnostic is there.
    out["harness_doctor_absent"] = runner.invoke(cli_app, ["harness", _DOCTOR]).exit_code == 2
    return out


# ---------------------------------------------------------------------------
# Battery + sealed receipt
# ---------------------------------------------------------------------------


def selftest_audit() -> dict[str, Any]:
    """Run every probe; returns ``name -> measured bool``."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_report_contract())
        out.update(_local_run_probes())
        out.update(_env_isolation_probes())
        out.update(_concurrency_probes())
        out.update(_state_dir_probes())
        out.update(_remote_probes())
        out.update(_gate_parity_probes())
        out.update(_bench_unit_probes())
        out.update(_cli_probes())
        out.update(_doctor_probes())
        return out


def selftest_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under selftest_audit.v1."""
    r = selftest_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "selftest_audit",
        "schema": "selftest_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process uvicorn on loopback + typer CliRunner; stub backends and engines",
            "not_executed": [
                "real remote deployments over public networks",
                "hosted BYOK upstreams",
                "cross-process selftest concurrency (in-process threads only)",
            ],
        },
        "interpretation": (
            "The deploy gate is honest end to end: a clean deployment "
            "passes with the pinned 16-check set (+1 under --state-dir, "
            "whose writes stay confined to the dir and produce no verdict "
            "artifact); a crashed check reports its exception class "
            "without masking later checks; empty reports are not ok; "
            "remote mode is provably read-only (GETs + advisory POSTs + "
            "one pinned 401 refusal) and leaves journals byte-identical; "
            "dead and 500-storm targets fail bounded with named checks; "
            "a missing key on a keyed deployment and a key on an "
            "unauth-gated deployment are both caught; ambient env — "
            "including the previously leaked MOONSHOT_API_KEY — is "
            "restored, a dead ambient BYOK URL can't substitute for the "
            "in-process stub, and concurrent runs no longer corrupt "
            "shared env; the bench records measured latencies and "
            "honest per-class error histograms, refuses bad params "
            "before spending, embeds the prompt as sha256 only, exits 1 "
            "on measured errors (dead target, wrong key, drained "
            "refusal storm), and seals a verifiable "
            "fx1_bench_result.v1 under --receipt; doctor emits presence "
            "flags that never carry secret values."
            if ok
            else f"SELFTEST AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(selftest_audit_bench(), indent=2, sort_keys=True))
