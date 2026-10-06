"""selftest — the harness's own golden-path smoke, zero config required.

``fx1 harness selftest`` boots a real uvicorn on loopback serving the
production ``create_app`` middleware stack plus a stub OpenAI-compatible
engine, then walks the whole consumption contract through the shipped
client: health, version negotiation, the auth gate, the error envelope,
commands, a gated BYOK completion, SSE streaming, idempotent replay, the
async job lifecycle, sealed-receipt verification, the drain latch, and
in-process parity with ``Fx1Harness`` — the same calls answered without
a socket.

With ``--state-dir`` it also proves durability: a second app instance
built on the same dir returns the first run's terminal job record.

``--remote URL`` flips to external mode: read-only/advisory checks only
(health, version, commands, score, gate pre-flight) — no model spend.

Every check reports ``{"name", "ok", "detail"}``; the command exits 0
only when all pass, 2 otherwise — usable as a deploy gate.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from typing import Any

__all__ = ["SelftestReport", "run_selftest"]


@dataclass
class _Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SelftestReport:
    mode: str
    checks: list[_Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and bool(self.checks)

    def note(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(_Check(name, ok, detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def _run(checks: list[_Check], name: str, fn: Any, expect: Any = None) -> None:
    """Record one check; ``fn()`` runs, result is compared to ``expect``
    when given, else any non-None value counts as a pass."""
    try:
        got = fn()
        ok = got == expect if expect is not None else bool(got)
        checks.append(_Check(name, ok, detail="" if ok else f"got {got!r}"))
    except Exception as exc:  # noqa: BLE001 — a check reports, never escapes
        checks.append(_Check(name, False, detail=f"{type(exc).__name__}: {exc}"))


def _wait_job(client: Any, job_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = dict(client.job_status(job_id))
        if st["status"] not in ("queued", "running"):
            return st
        time.sleep(0.05)
    return {"status": "timeout"}


def _remote_checks(report: SelftestReport, client: Any, remote: str, api_key: str) -> None:
    """Read-only checks safe against a live deployment."""
    from fx1.serve.contract import API_VERSION

    checks = report.checks
    _run(checks, "health", lambda: client.health())
    _run(
        checks,
        "version_matches_wire",
        lambda: client.server_version().get("api_version") == API_VERSION,
        True,
    )
    _run(checks, "commands", lambda: len(client.commands()) > 0, True)
    _run(
        checks,
        "score_advisory",
        lambda: client.score("the sharpe ratio was 9.9")[0].get("total", 0) < 0,
        True,
    )
    _run(
        checks,
        "gate_preflight",
        lambda: client.check_text("the sharpe ratio was 9.9").ok is False,
        True,
    )

    def _no_key_fails() -> bool:
        from fx1.serve.client import HarnessAuthError, HarnessClient

        bare = HarnessClient(remote, timeout_s=10.0)
        try:
            bare.complete([{"role": "user", "content": "x"}], backend="byok")
        except HarnessAuthError:
            return True
        except Exception:
            return False
        return False

    if api_key:
        _run(checks, "auth_gate_rejects_missing_key", _no_key_fails, True)


def run_selftest(
    remote: str | None = None,
    api_key: str | None = None,
    state_dir: str | None = None,
    timeout_s: float = 30.0,
) -> SelftestReport:
    """Run the golden-path smoke. ``remote`` targets a live deployment;
    otherwise a stub engine + the production app boot on loopback."""
    report = SelftestReport(mode="remote" if remote else "local")

    if remote is not None:
        from fx1.serve.client import HarnessClient

        client = HarnessClient(remote, api_key=api_key, timeout_s=timeout_s)
        _remote_checks(report, client, remote, api_key or "")
        return report

    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness
    from fx1.serve import api as api_mod
    from fx1.serve.backends import BackendNotConfiguredError
    from fx1.serve.client import HarnessClient
    from fx1.serve.contract import API_VERSION
    from fx1.serve.e2e_audit import _serve_uvicorn, _StubChat  # noqa: PLC0415

    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubChat)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    stub_port = int(stub.server_address[1])
    key = "selftest-key"  # a probe string, not a credential

    saved = {
        k: os.environ.get(k)
        for k in (
            "FX1_API_KEY",
            "FX1_BYOK_BASE_URL",
            "FX1_BYOK_API_KEY",
            "FX1_BYOK_MODEL",
            "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
            "MOONSHOT_API_KEY",
        )
    }
    os.environ.update(
        {
            "FX1_API_KEY": key,
            "FX1_BYOK_BASE_URL": f"http://127.0.0.1:{stub_port}/v1",
            "FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1",
            "FX1_BYOK_API_KEY": "stub-engine-key",
            "FX1_BYOK_MODEL": "stub-v0",
        }
    )
    os.environ.pop("MOONSHOT_API_KEY", None)

    # request logs are noise for a gate whose stdout is the JSON report —
    # silence the app's access logger for the run (and keep it out of any
    # captured stdout/stderr the caller may be mixing)
    api_logger = logging.getLogger("fx1.serve.api")
    prev_level = api_logger.level
    api_logger.setLevel(logging.CRITICAL)

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ran:" + " ".join(argv), ""

    server: Any = None
    server2: Any = None
    checks = report.checks
    try:
        app_kwargs: dict[str, Any] = {"harness": Harness(runner=fake_runner)}
        if state_dir is not None:
            app_kwargs["state_dir"] = state_dir
        app = api_mod.create_app(**app_kwargs)
        server, _t, port = _serve_uvicorn(app)
        _run(checks, "boot", lambda: bool(server.started), True)

        base = f"http://127.0.0.1:{port}"
        client = HarnessClient(base, api_key=key, timeout_s=timeout_s)
        msg = [{"role": "user", "content": "selftest"}]

        _run(
            checks,
            "health_byok",
            lambda: client.health().backends.get("byok") is True,
            True,
        )
        _run(
            checks,
            "version_matches_wire",
            lambda: client.server_version().get("api_version") == API_VERSION,
            True,
        )
        _run(checks, "commands", lambda: len(client.commands()) > 0, True)

        def _auth() -> bool:
            from fx1.serve.client import HarnessAuthError

            try:
                HarnessClient(base, timeout_s=timeout_s).complete(msg, backend="byok")
            except HarnessAuthError:
                return True
            except Exception:
                return False
            return False

        _run(checks, "auth_gate_rejects_missing_key", _auth, True)

        def _envelope() -> bool:
            try:
                client.complete(msg, backend="bogus")
            except ValueError:
                return True
            except Exception:
                return False
            return False

        _run(checks, "error_envelope_maps_valueerror", _envelope, True)
        _run(
            checks,
            "run_command",
            lambda: client.run(sorted(client.commands())[0]).exit_code == 0,
            True,
        )
        _run(
            checks,
            "complete_byok",
            lambda: client.complete(msg, backend="byok").content.startswith("stub:selftest"),
            True,
        )
        _run(
            checks,
            "stream_sse_reassembles",
            lambda: "".join(client.stream_complete(msg, backend="byok")).startswith(
                "stub:selftest"
            ),
            True,
        )

        def _idem() -> bool:
            k = "selftest-idem"
            a = client.submit_run("doctor", idempotency_key=k)
            b = client.submit_run("doctor", idempotency_key=k)
            return a == b

        _run(checks, "idempotent_replay", _idem, True)

        job_id = client.submit_run(sorted(client.commands())[0])
        final = _wait_job(client, job_id, timeout_s)
        checks.append(
            _Check(
                "job_lifecycle",
                final.get("status") == "succeeded",
                detail=str(final.get("status")),
            )
        )

        def _receipt() -> bool:
            rec = client.job_receipt(job_id)
            return client.verify_receipt(rec).valid

        _run(checks, "job_receipt_verifies", _receipt, True)

        def _score_gate() -> bool:
            s = client.score("the sharpe ratio was 9.9")[0]
            g = client.check_text("the sharpe ratio was 9.9")
            return s.get("total", 0) < 0 and g.ok is False

        _run(checks, "advisory_surfaces", _score_gate, True)

        # In-process twin must answer the same content without a socket.
        def _parity() -> bool:
            fx = Fx1Harness(
                harness=Harness(runner=fake_runner),
            )
            done = fx.complete(msg, backend="byok")
            return done.content.startswith("stub:selftest")

        _run(checks, "inprocess_parity", _parity, True)

        # Durability: restart the app on the same state dir; the job's
        # terminal record must still resolve.
        if state_dir is not None:
            server.should_exit = True
            deadline = time.monotonic() + 15.0
            while not _server_down(port) and time.monotonic() < deadline:
                time.sleep(0.05)
            app2 = api_mod.create_app(**app_kwargs)
            server2, _t2, port2 = _serve_uvicorn(app2)
            client = HarnessClient(f"http://127.0.0.1:{port2}", api_key=key, timeout_s=timeout_s)
            rec = client.job_status(job_id)
            checks.append(
                _Check(
                    "restart_recovers_job",
                    rec.get("status") == "succeeded",
                    detail=str(rec.get("status")),
                )
            )

        _run(
            checks,
            "drain_latches",
            lambda: client.drain().get("draining") is True,
            True,
        )

        def _drained() -> bool:
            try:
                client.run("doctor")
            except BackendNotConfiguredError:
                return True
            except Exception:
                return False
            return False

        _run(checks, "drained_rejects_work", _drained, True)
    finally:
        api_logger.setLevel(prev_level)
        for srv in (server, server2):
            if srv is not None:
                srv.should_exit = True
        stub.shutdown()
        stub.server_close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return report


def _server_down(port: int) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return False
    except OSError:
        return True
