"""Reuse one OpenAPI document across harness apps with the same routes.

``fx1.serve.api.create_app`` stamps response headers onto a schema that
FastAPI rebuilds from scratch. That rebuild is most of ``create_app``.
The route table does not vary between apps, so the first schema is kept
and every later app receives its own copy. ``_declare_response_headers``
mutates the copy it is given, which keeps a rate-limited app's headers
off an unlimited one.

Installed from the fx1_serve conftest. Production ``create_app`` is
unchanged until ``install`` runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import fastapi.applications as _fa

REAL_GET_OPENAPI = _fa.get_openapi
_BLOBS: dict[tuple[str, tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]], bytes] = {}
_INSTALLED = False


def _route_key(routes: Iterable[Any]) -> tuple[tuple[Any, ...], ...]:
    parts: list[tuple[Any, ...]] = []
    for route in routes:
        methods = getattr(route, "methods", None) or ()
        endpoint = getattr(route, "endpoint", None)
        parts.append(
            (
                getattr(route, "path", None),
                tuple(sorted(methods)),
                getattr(route, "name", None),
                getattr(endpoint, "__module__", None),
                getattr(endpoint, "__qualname__", None),
                type(route).__name__,
            )
        )
    return tuple(parts)


def _meta_key(kwargs: dict[str, Any]) -> str:
    meta = {key: value for key, value in kwargs.items() if key not in ("routes", "webhooks")}
    return json.dumps(meta, sort_keys=True, default=str)


def cached_get_openapi(**kwargs: Any) -> dict[str, Any]:
    key = (
        _meta_key(kwargs),
        _route_key(kwargs.get("routes") or ()),
        _route_key(kwargs.get("webhooks") or ()),
    )
    blob = _BLOBS.get(key)
    if blob is None:
        spec = REAL_GET_OPENAPI(**kwargs)
        blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        _BLOBS[key] = blob
    loaded: dict[str, Any] = json.loads(blob)
    return loaded


def install() -> None:
    """Point FastAPI's schema builder at :func:`cached_get_openapi`."""
    global _INSTALLED
    _fa.get_openapi = cached_get_openapi
    _INSTALLED = True
