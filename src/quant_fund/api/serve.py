"""Managed process entry for the lab API and the read-only research API.

Validates :class:`ServiceSettings`, emits a startup JSON line, installs the
optional metrics hooks, and serves until SIGTERM. SIGTERM sets the drain
latch and then stops accepting connections. SIGUSR1 sets the same latch and
leaves the socket open so a proxy can take the replica out of rotation first.
An owner replica holds the data-directory write lock before it binds.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
from contextlib import suppress
from typing import Any

import uvicorn

from quant_fund.api.json_logging import emit_service_event, json_log_config
from quant_fund.api.lifecycle import arm, begin_drain, disarm
from quant_fund.api.service_config import ServiceConfigError, ServiceSettings
from quant_fund.utils.write_owner import WriteOwnerError, claim_write_lock


def _on_drain_signal(_signum: int, _frame: object) -> None:
    """SIGUSR1 stops advertising readiness and leaves the socket open."""
    begin_drain()


def install_drain_signal() -> None:
    """Arm SIGUSR1 so a proxy can remove this replica before it stops."""
    signal.signal(signal.SIGUSR1, _on_drain_signal)


def _claim_write_owner() -> None:
    try:
        claim_write_lock()
    except WriteOwnerError as exc:
        raise SystemExit(str(exc)) from exc


class _DrainServer(uvicorn.Server):
    """SIGTERM/SIGINT drains readiness before the server stops accepting.

    SIGUSR1 is armed on the running loop so the drain latch flips while the
    socket stays open. ``signal.signal`` alone can miss the loop.
    """

    async def serve(self, sockets: Any = None) -> None:
        with suppress(RuntimeError, NotImplementedError, ValueError):
            asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, begin_drain)
        await super().serve(sockets)

    def handle_exit(self, sig: int, frame: Any) -> None:
        begin_drain()
        with suppress(OSError, TypeError, ValueError):
            emit_service_event("service_draining", signal=sig)
        super().handle_exit(sig, frame)


def run_uvicorn(
    application: Any,
    *,
    host: str,
    port: int,
    log_level: str,
    shutdown_grace_s: int,
) -> None:
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        reload=False,
        log_config=json_log_config(log_level),
        log_level=log_level.lower(),
        timeout_graceful_shutdown=shutdown_grace_s,
        server_header=False,
        proxy_headers=True,
    )
    _DrainServer(config).run()


def _serve(settings: ServiceSettings, application: Any) -> None:
    emit_service_event("service_start", **settings.public_dict())
    os.environ["DIPCATCHER_METRICS_KEY_ENV"] = (
        "QUANT_API_KEY" if settings.service == "lab" else "RESEARCH_API_KEY"
    )
    _claim_write_owner()
    install_drain_signal()
    arm(settings)
    try:
        from quant_fund.api.ops_metrics import install_ops_sampler
        from quant_fund.observe.install import install_passive_hooks

        install_ops_sampler()
        install_passive_hooks()
        run_uvicorn(
            application,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            shutdown_grace_s=settings.shutdown_grace_s,
        )
    finally:
        from quant_fund.observe.metrics import clear_ops_sampler
        from quant_fund.observe.serve import stop_metrics_server

        stop_metrics_server()
        clear_ops_sampler()
        disarm()


def serve_lab(host: str | None = None, port: int | None = None) -> None:
    _serve(ServiceSettings.for_lab(host, port), "quant_fund.api.app:app")


def serve_research(application: Any | None = None) -> None:
    if application is None:
        from quant_fund.api.research_api import app as research_app

        application = research_app
    _serve(ServiceSettings.for_research(), application)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quant_fund.api.serve")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        serve_lab(args.host, args.port)
    except ServiceConfigError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
