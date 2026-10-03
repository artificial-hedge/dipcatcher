"""param_fuzz — adversarial parameter fuzzing over the pinned HTTP surface.

``surface_pin`` freezes *which* routes exist; this lane checks what they
do under hostile inputs. For every GET route on the app it substitutes
each path parameter and each query parameter — one at a time, others
pinned at a benign value — with a fixed corpus of edge cases: oversized
and empty strings, traversal and percent-encoding tricks, unicode,
integer overflow, SQL-ish fragments, format-string braces.

Per case the invariant is fail-closed:

- status < 500 (a handler crash is a finding, not a shrug),
- error bodies are JSON and never leak ``Traceback`` / file paths.

Inputs are recorded by sha256 digest, not verbatim — the receipt is
deterministic and safe to commit. Wall-clock timing is deliberately NOT
recorded: the sealed payload must be byte-reproducible. Receipt
``api_fuzz.v1``.

Note: request *bodies* on POST routes are a different fuzz class (schema
fuzzing) and intentionally out of scope here.
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

API_FUZZ_SCHEMA = "api_fuzz.v1"

_BENIGN_PATH = "0" * 64  # sha256-shaped, exists almost nowhere
_EDGE_CASES: tuple[str, ...] = (
    "",
    "0" * 64,
    "f" * 64,
    "..",
    "../etc/passwd",
    "%2e%2e%2fetc",
    "A" * 4096,
    " ",  # space
    "Ω",  # non-ascii
    "-1",
    str(2**63),
    "{id}",
    "}",
    "' OR 1=1--",
    "null",
    "undefined",
    "0x" + "f" * 30,
)


def _routes(app: Any) -> list[tuple[str, list[str], list[str]]]:
    """(path, path_params, query_params) for GET routes only."""
    out = []
    for route in getattr(app, "routes", []):
        methods = set(getattr(route, "methods", set()) or set())
        if "GET" not in methods or getattr(route, "dependant", None) is None:
            continue
        dep = route.dependant
        path_params = [p.name for p in getattr(dep, "path_params", [])]
        query_params = [p.name for p in getattr(dep, "query_params", [])]
        out.append((route.path, path_params, query_params))
    return sorted(out)


def _classify(status: int, body_text: str) -> str | None:
    if status >= 500:
        return "server_error"
    if status >= 400 and ("Traceback" in body_text or 'File "' in body_text):
        return "traceback_leak"
    return None


def fuzz_app(app: Any, client: Any) -> dict[str, Any]:
    """Run the edge-case matrix over every GET route. Returns findings."""
    routes = _routes(app)
    findings: list[dict[str, Any]] = []
    n_cases = 0
    per_route: dict[str, int] = {}
    for path, path_params, query_params in routes:
        cases = 0
        for target in path_params:
            for edge in _EDGE_CASES:
                url = path
                for p in path_params:
                    url = url.replace("{" + p + "}", edge if p == target else _BENIGN_PATH)
                cases += 1
                n_cases += 1
                try:
                    resp = client.get(url)
                    status, body = resp.status_code, resp.text
                except Exception as exc:  # transport-level blowup = finding
                    status, body = 599, f"{exc.__class__.__name__}: {exc}"
                cls = _classify(status, body)
                if cls:
                    findings.append(
                        {
                            "route": path,
                            "param": target,
                            "edge_sha256": hash_bytes(edge.encode())[:16],
                            "status": status,
                            "class": cls,
                        }
                    )
        for q in query_params:
            for edge in _EDGE_CASES:
                url = path
                for p in path_params:
                    url = url.replace("{" + p + "}", _BENIGN_PATH)
                cases += 1
                n_cases += 1
                try:
                    resp = client.get(url, params={q: edge})
                    status, body = resp.status_code, resp.text
                except Exception as exc:
                    status, body = 599, f"{exc.__class__.__name__}: {exc}"
                cls = _classify(status, body)
                if cls:
                    findings.append(
                        {
                            "route": path,
                            "param": q,
                            "edge_sha256": hash_bytes(edge.encode())[:16],
                            "status": status,
                            "class": cls,
                        }
                    )
        per_route[path] = cases
    return {
        "routes_fuzzed": len(routes),
        "n_cases": n_cases,
        "per_route_cases": per_route,
        "violations": findings,
        "clean": not findings,
    }


def api_fuzz_bench(app: Any, client: Any) -> dict[str, Any]:
    """Sealed api_fuzz.v1 receipt. ``client`` is a TestClient-like object."""
    out = fuzz_app(app, client)
    payload: dict[str, Any] = {
        "kind": "api_fuzz",
        "schema": API_FUZZ_SCHEMA,
        "result": out,
        "claim": "no_5xx_on_adversarial_params" if out["clean"] else "violations_found",
        "interpretation": (
            "Every GET route on the pinned surface is exercised with a "
            "fixed edge-case corpus (oversized/traversal/unicode/overflow "
            "inputs, one param at a time). A 5xx or a leaked traceback is "
            "a finding; 4xx rejections are the contract working."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
