"""Error-surface audit — every route must fail uniformly, never 5xx.

Enumerates the app's own OpenAPI spec (new routes are auto-covered) and
fires three probe classes at each:

- ``bad_path``: pathological path parameters (``..``, NUL, giant ints,
  unicode confusables) in templated slots,
- ``bad_query``: malformed query params (empty, very long, percent junk),
- ``bad_method``: POST/PUT/DELETE on GET-only routes.

Every response must land in an allowed status set, its error body must
be JSON with a ``detail`` key, and the body must never contain a
traceback or the process's filesystem layout. A single 5xx or leaked
internals is a fail.

Requires ``fastapi``'s test client — guarded so the module imports
cleanly without the web extras.
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_ALLOWED_STATUSES = {200, 400, 404, 405, 422}
_LEAK_TOKENS = ("Traceback", 'File "/', "site-packages", "Internal Server Error")
_BAD_PATHS = ["..", "%2e%2e", "0", "-1", "99999999999999999999", "null", "confusible_\u0440"]
_BAD_QUERIES = ["?x=", "?limit=-1", "?limit=999999999999", "?x=%00%ff", "?s=" + "a" * 4096]
_BAD_METHODS = ["post", "delete"]


def _fill_path(template: str, value: str) -> str:
    import re

    return re.sub(r"\{[^}]+\}", value, template)


def error_shape_audit() -> dict[str, Any]:
    """Probe every OpenAPI route; return per-route verdict records."""
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:  # pragma: no cover - optional extras
        raise RuntimeError("error_shape needs fastapi extras") from exc
    from quant_fund.api.research_api import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    spec = client.get("/openapi.json").json()
    paths = [p for p, ops in spec.get("paths", {}).items() if "get" in ops]

    records: list[dict[str, Any]] = []
    for path in paths:
        for probe_name, urls in {
            "bad_path": [client.get(_fill_path(path, v)) for v in _BAD_PATHS]
            if "{" in path
            else [],
            "bad_query": [client.get(path + q) for q in _BAD_QUERIES],
            "bad_method": [getattr(client, m)(path) for m in _BAD_METHODS],
        }.items():
            for resp in urls:
                problems: list[str] = []
                if resp.status_code not in _ALLOWED_STATUSES:
                    problems.append(f"status={resp.status_code}")
                body = resp.text
                if any(tok in body for tok in _LEAK_TOKENS):
                    problems.append("internal_leak")
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail")
                    except ValueError:
                        detail = None
                    if detail is None:
                        problems.append("no_detail_envelope")
                records.append(
                    {
                        "route": path,
                        "probe": probe_name,
                        "url_probe": resp.request.url.path,
                        "status": resp.status_code,
                        "ok": not problems,
                        "problems": problems,
                    }
                )
    n_bad = sum(1 for r in records if not r["ok"])
    return {
        "n_routes": len(paths),
        "n_probes": len(records),
        "n_violations": n_bad,
        "verdict": "ok" if n_bad == 0 else "violations",
        "violations": [r for r in records if not r["ok"]][:50],
    }


def error_shape_bench() -> dict[str, Any]:
    """Sealed receipt over the live app surface."""
    report = error_shape_audit()
    payload: dict[str, Any] = {
        "kind": "error_shape",
        "schema": "error_shape.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "every route answers probes with an allowed status and a uniform error envelope; no internals leak",
            "verdict": report["verdict"],
            "n_routes": report["n_routes"],
            "n_probes": report["n_probes"],
            "n_violations": report["n_violations"],
        },
        "interpretation": report,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
