"""selftest_audit — the golden-path smoke's own contract battery.

``fx1 harness selftest`` is the deploy gate: it must be green on a
healthy deployment AND honest about partial coverage. This battery pins
the machinery *beneath* the golden walk:

- *report semantics* — ``ok`` requires nonempty all-true checks,
  ``note()``/``as_dict()`` shape stability, detail propagation.
- *_run discipline* — expect-comparison vs truthiness, exception
  capture into ``{Type}: {msg}`` details, checks never escape.
- *_wait_job* — terminal statuses return immediately; a running job
  past the deadline reports ``{"status": "timeout"}`` instead of
  hanging.
- *_server_down* — live port → False, closed port → True.
- *remote mode* — unreachable target records failures (never raises),
  mode/report honesty; and remote mode against a real in-process app
  runs the read-only battery including the missing-key auth-gate
  check.
- *local mode* — the full golden path actually passes, with the exact
  expected check set, and every environment variable and logger level
  it touched is restored (the audit found and the lane fixes the
  ``FX1_BYOK_ALLOW_PRIVATE_NETWORKS``/``MOONSHOT_API_KEY`` restore
  gaps).

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fx1.selftest import (
    SelftestReport,
    _Check,
    _run,
    _server_down,
    _wait_job,
    run_selftest,
)

__all__ = ["selftest_audit", "selftest_audit_bench"]


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


class _FakeClient:
    def __init__(self, status: str) -> None:
        self._status = status

    def job_status(self, job_id: str) -> dict[str, Any]:
        return {"status": self._status, "job_id": job_id}


def _probe_report() -> dict[str, bool]:
    out: dict[str, bool] = {}
    empty = SelftestReport(mode="local")
    out["rp_empty_not_ok"] = empty.ok is False
    r = SelftestReport(mode="remote")
    r.note("a", True, "d")
    out["rp_note_appends"] = len(r.checks) == 1
    out["rp_ok_all_true"] = r.ok is True
    r.note("b", False)
    out["rp_mixed_not_ok"] = r.ok is False
    d = r.as_dict()
    out["rp_as_dict_shape"] = (
        d["mode"] == "remote"
        and d["ok"] is False
        and d["checks"][0] == {"name": "a", "ok": True, "detail": "d"}
        and d["checks"][1]["ok"] is False
    )
    out["rp_check_dataclass"] = _Check("n", True, "x").detail == "x"
    return out


def _probe_run_helper() -> dict[str, bool]:
    out: dict[str, bool] = {}
    checks: list[_Check] = []
    _run(checks, "expect_match", lambda: 4, 4)
    _run(checks, "expect_mismatch", lambda: 5, 4)
    _run(checks, "truthy", lambda: [1])
    _run(checks, "falsy", lambda: [])
    _run(checks, "exc", lambda: 1 / 0)
    by_name = {c.name: c for c in checks}
    out["rh_expect_match"] = by_name["expect_match"].ok is True
    out["rh_expect_mismatch"] = (
        by_name["expect_mismatch"].ok is False and "got" in by_name["expect_mismatch"].detail
    )
    out["rh_truthy"] = by_name["truthy"].ok is True
    out["rh_falsy"] = by_name["falsy"].ok is False and by_name["falsy"].detail != ""
    out["rh_exc_captured"] = by_name["exc"].ok is False and by_name["exc"].detail.startswith(
        "ZeroDivisionError"
    )
    out["rh_never_raises"] = len(checks) == 5
    return out


def _probe_wait_and_sockets() -> dict[str, bool]:
    out: dict[str, bool] = {}
    done = _wait_job(_FakeClient("succeeded"), "j1", timeout_s=1.0)
    out["wj_terminal_return"] = done["status"] == "succeeded"
    failed = _wait_job(_FakeClient("failed"), "j2", timeout_s=1.0)
    out["wj_failed_terminal"] = failed["status"] == "failed"
    t0 = time.monotonic()
    timed = _wait_job(_FakeClient("running"), "j3", timeout_s=0.3)
    out["wj_timeout"] = timed == {"status": "timeout"} and time.monotonic() - t0 < 5.0
    out["sd_closed_port"] = _server_down(1) is True
    import socket

    live = socket.socket()
    live.bind(("127.0.0.1", 0))
    live.listen(1)
    try:
        out["sd_live_port"] = _server_down(int(live.getsockname()[1])) is False
    finally:
        live.close()
    return out


@functools.lru_cache(maxsize=1)
def _golden() -> SelftestReport:
    return run_selftest()


def _probe_remote_unreachable() -> dict[str, bool]:
    out: dict[str, bool] = {}
    report = run_selftest(remote="http://127.0.0.1:1", api_key="k", timeout_s=2.0)
    out["ru_mode"] = report.mode == "remote"
    out["ru_not_ok"] = report.ok is False
    out["ru_checks_recorded"] = len(report.checks) >= 4
    out["ru_details_honest"] = all((c.ok is True) or c.detail != "" for c in report.checks)
    return out


def _probe_remote_live() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.harness import Harness
    from fx1.serve import api as api_mod
    from fx1.serve.e2e_audit import _serve_uvicorn, _StubChat  # noqa: PLC0415

    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubChat)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    port_stub = int(stub.server_address[1])
    sentinel = {
        "FX1_API_KEY": "example-remote-key",
        "FX1_BYOK_BASE_URL": f"http://127.0.0.1:{port_stub}/v1",
        "FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1",
        "FX1_BYOK_API_KEY": "stub-engine-key",
        "FX1_BYOK_MODEL": "stub-v0",
    }
    saved = {k: os.environ.get(k) for k in sentinel}
    server = None
    try:
        os.environ.update(sentinel)
        os.environ.pop("MOONSHOT_API_KEY", None)

        def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            return 0, "ran", ""

        app = api_mod.create_app(harness=Harness(runner=fake_runner))
        server, _t, port = _serve_uvicorn(app)
        base = f"http://127.0.0.1:{port}"
        report = run_selftest(remote=base, api_key="example-remote-key", timeout_s=15.0)
        names = {c.name: c.ok for c in report.checks}
        out["rl_mode"] = report.mode == "remote"
        out["rl_all_green"] = report.ok is True
        out["rl_expected_set"] = {
            "health",
            "version_matches_wire",
            "commands",
            "score_advisory",
            "gate_preflight",
            "auth_gate_rejects_missing_key",
        } <= set(names)
        out["rl_auth_check_ran"] = names.get("auth_gate_rejects_missing_key") is True
    finally:
        if server is not None:
            server.should_exit = True
        stub.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _probe_local_golden() -> dict[str, bool]:
    out: dict[str, bool] = {}
    api_logger = logging.getLogger("fx1.serve.api")
    before_level = api_logger.level
    sentinel_env = {
        "FX1_API_KEY": "audit-sentinel-key",
        "FX1_BYOK_BASE_URL": "https://sentinel.invalid/v1",
        "FX1_BYOK_API_KEY": "audit-sentinel-byok",
        "FX1_BYOK_MODEL": "sentinel-model",
        "FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "0",
        "MOONSHOT_API_KEY": "audit-sentinel-moon",
    }
    saved = {k: os.environ.get(k) for k in sentinel_env}
    try:
        os.environ.update(sentinel_env)
        report = _golden()
        names = {c.name: c.ok for c in report.checks}
        out["lg_all_green"] = report.ok is True
        out["lg_mode"] = report.mode == "local"
        out["lg_expected_set"] = {
            "boot",
            "health_byok",
            "version_matches_wire",
            "commands",
            "auth_gate_rejects_missing_key",
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
        } <= set(names)
        out["lg_no_restart_without_state"] = "restart_recovers_job" not in names
        out["lg_env_restored"] = all(os.environ.get(k) == v for k, v in sentinel_env.items())
        out["lg_logger_restored"] = api_logger.level == before_level
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _probe_state_dir(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    saved = {
        k: os.environ.get(k)
        for k in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_BYOK_ALLOW_PRIVATE_NETWORKS")
    }
    try:
        os.environ["FX1_API_KEY"] = "audit-state-key"
        report = run_selftest(state_dir=str(tmp / "state"), timeout_s=30.0)
        names = {c.name: c.ok for c in report.checks}
        out["sd_all_green"] = report.ok is True
        out["sd_restart_check"] = names.get("restart_recovers_job") is True
        out["sd_env_restored"] = os.environ.get("FX1_API_KEY") == ("audit-state-key")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def selftest_audit() -> dict[str, bool]:
    """Every selftest contract as booleans."""
    with tempfile.TemporaryDirectory() as tmp:
        out: dict[str, bool] = {}
        out.update(_probe_report())
        out.update(_probe_run_helper())
        out.update(_probe_wait_and_sockets())
        out.update(_probe_remote_unreachable())
        out.update(_probe_remote_live())
        out.update(_probe_local_golden())
        out.update(_probe_state_dir(Path(tmp).resolve()))
    return out


def selftest_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the selftest battery."""
    r = selftest_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "selftest_audit",
        "schema": "selftest_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process uvicorn + stub engine on loopback; remote mode against a self-hosted app",
            "not_verified": [
                "selftest against a production deployment",
                "CLI flag surface (covered by tests/fx1/test_selftest.py)",
            ],
        },
        "interpretation": (
            "Selftest machinery holds: reports are honest (empty is not "
            "ok), check capture never escapes, job polling reports "
            "timeout instead of hanging, remote mode is read-only and "
            "verifies the auth gate, the local golden path passes "
            "end-to-end, and every env var + logger level it touches is "
            "restored afterward."
            if ok
            else f"SELFTEST AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(selftest_audit_bench(), indent=2, sort_keys=True))
