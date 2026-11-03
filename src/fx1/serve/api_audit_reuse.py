"""Reuse stateless OpenAI-compatible apps inside one api-audit call.

A shared app keeps idempotency and key stores. Probes that mint keys
against a fresh app must not hit ``_CleanBackend``.
"""

from __future__ import annotations

import os
from typing import Any

# These constructors look stateless, but a shared app keeps their stores.
_REUSE_DENY_CLASSES = frozenset({"_CleanBackend"})


def isolated_client(api_key: str | None, env_names: tuple[str, ...]) -> tuple[Any, Any]:
    """Build one TestClient with a hermetic runner and a restored env."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    saved = {name: os.environ.get(name) for name in env_names}
    try:
        if api_key is None:
            os.environ.pop("FX1_API_KEY", None)
        else:
            os.environ["FX1_API_KEY"] = api_key
        app = api_mod.create_app(harness=Harness(runner=fake_runner))
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _oi_lambda_key(resolver: Any) -> tuple[Any, ...] | None:
    """Identity of a lambda whose closures are classes, not instances.

    ``lambda *a, **k: _OiBackend()`` at two call sites shares one app.
    ``_hdr_resolver`` is not a lambda, and a lambda that closes over a live
    backend instance does not either. ``_CleanBackend`` is denied: key
    probes need the store those apps start with.
    """
    code = getattr(resolver, "__code__", None)
    if code is None or code.co_name != "<lambda>":
        return None
    names: list[str] = []
    cells: list[int] = []
    for cell in resolver.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            return None
        if not isinstance(value, type) or value.__name__ in _REUSE_DENY_CLASSES:
            return None
        names.append(value.__name__)
        cells.append(id(value))
    if not names:
        if not code.co_names or any(name in _REUSE_DENY_CLASSES for name in code.co_names):
            return None
        names.extend(code.co_names)
    try:
        consts = tuple(code.co_consts)
        hash(consts)
    except TypeError:
        return None
    return (code.co_code, tuple(names), consts, tuple(cells))


class _OiAppReuse:
    """Reuse apps built by identical stateless backend lambdas.

    The cache lives for one ``api_audit`` call. Constructor kwargs and the
    ``FX1_*`` environment are part of the key, so a rate-limited app does
    not satisfy an unlimited one.
    """

    def __enter__(self) -> _OiAppReuse:
        import fx1.serve.api as api_mod

        self._mod = api_mod
        self._real = api_mod.create_app
        cache: dict[tuple[Any, ...], Any] = {}

        def _create(*args: Any, **kwargs: Any) -> Any:
            resolver = kwargs.get("backend_resolver")
            ident = _oi_lambda_key(resolver) if resolver is not None else None
            if args or ident is None:
                return self._real(*args, **kwargs)
            extra: list[tuple[str, Any]] = []
            for name, value in sorted(kwargs.items()):
                if name == "backend_resolver":
                    continue
                try:
                    hash(value)
                except TypeError:
                    return self._real(*args, **kwargs)
                extra.append((name, value))
            env = tuple(
                sorted(
                    (key, os.environ.get(key))
                    for key in os.environ
                    if key.startswith("FX1_") or key == "MOONSHOT_API_KEY"
                )
            )
            key = (ident, tuple(extra), env)
            app = cache.get(key)
            if app is None:
                app = self._real(*args, **kwargs)
                cache[key] = app
            return app

        api_mod.create_app = _create  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: object) -> None:
        self._mod.create_app = self._real
