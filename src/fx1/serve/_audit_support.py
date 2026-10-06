"""Canonical machinery for fx1 serve audit batteries.

The first generation of ``*_audit`` modules each vendored a private copy
of this scaffold — the serialized environment guard, the per-scope
resource stack, the isolated ``TestClient`` factory, and the
``HarnessClient`` transport adapter. New batteries import the canonical
versions here instead of vendoring a tenth copy; the historical modules
keep theirs (their sealed receipts predate this file and pin their own
provenance).
"""

from __future__ import annotations

import os
import tempfile
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["audit_scope", "scoped_resources", "scoped_tmpdir", "tc_transport", "test_client"]

_serial = threading.Lock()
_stack_var: ContextVar[ExitStack] = ContextVar("fx1_serve_audit_stack")

_API_KEY_ENV = "FX1_API_KEY"


def _guarded_env() -> dict[str, str]:
    """Ambient harness configuration the scope takes over."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith("FX1_") or key == "MOONSHOT_API_KEY"
    }


@contextmanager
def audit_scope() -> Iterator[ExitStack]:
    """Serialized, environment-isolated scope for one audit battery.

    Clearing ``FX1_*`` and ``MOONSHOT_API_KEY`` inside the scope keeps a
    leaky developer shell out of the measured app; everything is
    restored on exit, faults included. The lock makes concurrent
    batteries in one process impossible by construction — the
    environment is process-wide, so batteries run serialized here or in
    dedicated processes.
    """
    with _serial:
        stashed = _guarded_env()
        for key in stashed:
            os.environ.pop(key, None)
        stack = ExitStack()
        token = _stack_var.set(stack)
        try:
            with stack:
                yield stack
        finally:
            _stack_var.reset(token)
            for key in _guarded_env():
                os.environ.pop(key, None)
            os.environ.update(stashed)


def scoped_resources() -> ExitStack:
    """The innermost ``audit_scope`` ExitStack — register callbacks and
    ``enter_context`` for anything the scope must tear down."""
    return _stack_var.get()


def scoped_tmpdir(prefix: str = "fx1_audit_") -> Path:
    """A ``TemporaryDirectory`` owned by the active scope."""
    return Path(_stack_var.get().enter_context(tempfile.TemporaryDirectory(prefix=prefix)))


def test_client(
    backends: dict[str, Callable[[], Any]],
    api_key: str | None = None,
    **app_kwargs: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) over a fresh, fully isolated app.

    Each call builds receipts/state/ft dirs under a scoped tmpdir,
    resolves backend names through ``backends`` zero-arg factories, and
    registers teardown (jobs executor, client.close) on the audit
    scope's stack. ``app_kwargs`` lands verbatim on ``create_app``
    (``state_dir``, ``rate_limit_rps``, ``max_inflight``, ...).
    """
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def _noop_runner(_argv: list[str], _timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    isolated = scoped_tmpdir()
    (isolated / "receipts").mkdir()
    prior_key = os.environ.pop(_API_KEY_ENV, None)
    try:
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=_noop_runner),
            backend_resolver=lambda name, *_a, **_k: backends[name](),
            receipts_dir=isolated / "receipts",
            ft_dir=isolated / "fine_tuning",
            state_dir=app_kwargs.pop("state_dir", isolated / "state"),
            **app_kwargs,
        )
        stack = scoped_resources()
        stack.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = _TC(app, raise_server_exceptions=False)
        stack.callback(client.close)
        stack.enter_context(client)
        return client, api_mod
    finally:
        if prior_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = prior_key


def tc_transport(client: TestClient) -> Callable[..., tuple[int, Any, bytes]]:
    """HarnessClient's transport contract adapted to a TestClient —
    the client/SDK legs twin the wire without sockets."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        u = urllib.parse.urlparse(url)
        target = u.path + ("?" + u.query if u.query else "")
        if method == "GET":
            resp = client.get(target, headers=headers)
        elif method == "DELETE":
            resp = client.delete(target, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(target, content=payload, headers=headers)
        else:
            resp = client.post(target, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send
