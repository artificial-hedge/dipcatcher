"""Process lifecycle: drain latch, readiness block, and ASGI lifespan.

Importing this module does not bind a socket or read configuration files.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from quant_fund.api.kill_file import inspect_kill_switch

_draining = threading.Event()
_managed: ManagedRuntime | None = None


class ManagedRuntime(Protocol):
    @property
    def service(self) -> str: ...

    @property
    def observe(self) -> bool: ...

    @property
    def alert_interval_s(self) -> int: ...

    @property
    def slo_path(self) -> str | None: ...

    def public_dict(self) -> dict[str, str | int | bool | None]: ...


def arm(settings: ManagedRuntime) -> None:
    """Mark this process as the managed service and clear a stale drain."""
    global _managed
    _managed = settings
    _draining.clear()


def disarm() -> None:
    global _managed
    _managed = None


def current_settings() -> ManagedRuntime | None:
    return _managed


def begin_drain() -> None:
    """Stop advertising readiness. Idempotent."""
    _draining.set()


def reset_runtime_state() -> None:
    """Test helper. Clears the drain latch and the managed-service pointer."""
    disarm()
    _draining.clear()


def acceptance_block() -> str | None:
    """Why the process must not receive work. ``None`` means it may.

    ``draining`` wins over the kill-switch file. The file is re-read on every
    call so an operator can halt without a restart. The returned string is a
    fixed code, never the file body.
    """
    if _draining.is_set():
        return "draining"
    raw = os.environ.get("DIPCATCHER_KILL_SWITCH_PATH", "").strip()
    if not raw:
        return None
    reading = inspect_kill_switch(Path(raw))
    if reading.problem:
        return reading.problem
    if reading.blocking and reading.state:
        return f"kill_switch:{reading.state}"
    return None


def _stop_metrics() -> None:
    from quant_fund.observe.serve import stop_metrics_server

    stop_metrics_server()


def _emit_shutdown(settings: ManagedRuntime) -> None:
    from quant_fund.api.json_logging import emit_service_event

    try:
        emit_service_event("service_shutdown", **settings.public_dict())
    except (OSError, TypeError, ValueError):
        return


@asynccontextmanager
async def service_lifespan(_app: object) -> AsyncIterator[None]:
    """Start the alert loop after bind prep, and drain on the way down.

    Uvicorn runs this shutdown hook after the graceful connection wait. The
    SIGTERM handler drains earlier so ``/ready`` fails while connections finish.
    Unmanaged imports (the test client, a bare ASGI load) do nothing.
    """
    settings = current_settings()
    monitor: object | None = None
    if settings is not None and settings.observe:
        from quant_fund.api.alerts import AlertMonitor

        started = AlertMonitor(settings)
        started.start()
        monitor = started
    try:
        yield
    finally:
        if settings is not None:
            begin_drain()
        if monitor is not None:
            stop = getattr(monitor, "stop", None)
            if stop is not None:
                stop()
        if settings is not None:
            _stop_metrics()
            _emit_shutdown(settings)
