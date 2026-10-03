"""surface_pin — pin the HTTP surface; drift is a test failure, not a surprise.

The public API is a contract: endpoints, methods, and declared request
parameters. Today nothing watches it — a route can be renamed, a method
dropped, or a required parameter added without any gate noticing.

``surface()`` enumerates the FastAPI app's routes into a canonical map:
``{path: {method: {"params": [...], "response_fields": [...]}}}`` where
``params`` are the handler's declared non-body arguments (name +
required flag) and ``response_fields`` are the response model's field
names when declared. ``diff`` against the committed pin reports
``route_removed`` / ``route_added`` / ``method_removed`` /
``param_changed`` / ``response_fields_changed`` — additions are visible
but only *breaking* deltas (removals, new required params, response
shape changes) make ``breaking=True``.

The pin file lives at ``quality/api_surface.json``; regenerate after an
intentional surface change.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

# params injected by the framework or the auth middleware — not contract
_SKIP_PARAMS = {"request", "response", "self", "cls"}


def _route_entry(route: Any) -> dict[str, Any] | None:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return None
    endpoint = getattr(route, "endpoint", None)
    # framework-served routes (/docs, /openapi.json, /redoc) aren't our
    # contract — they move with the FastAPI version, not our code
    if endpoint is not None and str(getattr(endpoint, "__module__", "")).startswith("fastapi"):
        return None
    params: list[dict[str, Any]] = []
    if endpoint is not None:
        for name, p in inspect.signature(endpoint).parameters.items():
            if name in _SKIP_PARAMS or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            params.append({"name": name, "required": p.default is inspect.Parameter.empty})
    response_fields: list[str] = []
    model = getattr(route, "response_model", None)
    fields = getattr(model, "model_fields", None)
    if isinstance(fields, dict):
        response_fields = sorted(fields)
    entry: dict[str, Any] = {}
    for m in sorted(methods):
        if m in {"HEAD", "OPTIONS"}:
            continue
        entry[m] = {"params": sorted(params, key=lambda x: x["name"])}
        if response_fields:
            entry[m]["response_fields"] = response_fields
    return {"path": path, "methods": entry}


def surface(app: Any) -> dict[str, Any]:
    """Canonical surface map {path: {method: {params, response_fields}}}."""
    out: dict[str, Any] = {}
    for route in app.routes:
        e = _route_entry(route)
        if e is None:
            continue
        out[e["path"]] = e["methods"]
    return dict(sorted(out.items()))


def diff(pinned: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Compare a pinned surface against the live one."""
    issues: list[str] = []
    for path in sorted(set(pinned) - set(live)):
        issues.append(f"route_removed:{path}")
    for path in sorted(set(live) - set(pinned)):
        issues.append(f"route_added:{path}")
    for path in sorted(set(pinned) & set(live)):
        p_methods, l_methods = pinned[path], live[path]
        for m in sorted(set(p_methods) - set(l_methods)):
            issues.append(f"method_removed:{m} {path}")
        for m in sorted(set(l_methods) - set(p_methods)):
            issues.append(f"method_added:{m} {path}")
        for m in sorted(set(p_methods) & set(l_methods)):
            pp = {p["name"]: p["required"] for p in p_methods[m].get("params", [])}
            lp = {p["name"]: p["required"] for p in l_methods[m].get("params", [])}
            if pp != lp:
                issues.append(f"param_changed:{m} {path}: {sorted(pp)} -> {sorted(lp)}")
            pf = p_methods[m].get("response_fields")
            lf = l_methods[m].get("response_fields")
            if pf is not None and lf is not None and pf != lf:
                issues.append(f"response_fields_changed:{m} {path}")
    breaking = any(
        i.startswith(
            ("route_removed", "method_removed", "param_changed", "response_fields_changed")
        )
        for i in issues
    )
    return {"issues": issues, "breaking": breaking, "n_issues": len(issues)}


def load_pin(path: Path | str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data


def check(app: Any, pin_path: Path | str) -> dict[str, Any]:
    """Live app vs committed pin; ``ok`` = no breaking drift."""
    d = diff(load_pin(pin_path), surface(app))
    d["ok"] = not d["breaking"]
    return d


__all__ = ["check", "diff", "load_pin", "surface"]
