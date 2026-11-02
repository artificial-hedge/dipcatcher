"""Managed process entry for the lab API and the read-only research API.

Validates :class:`ServiceSettings`, emits a startup JSON line, installs the
optional metrics hooks, and serves until SIGTERM. The signal handler sets the
drain latch before uvicorn stops accepting connections.
"""

from __future__ import annotations

import argparse
import os
from contextlib import suppress
from typing import Any

import uvicorn

from quant_fund.api.json_logging import emit_service_event, json_log_config
from quant_fund.api.lifecycle import arm, begin_drain, disarm
from quant_fund.api.service_config import ServiceConfigError, ServiceSettings


class _DrainServer(uvicorn.Server):
    """SIGTERM/SIGINT drains readiness before the server stops accepting."""

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
