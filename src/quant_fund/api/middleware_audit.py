"""middleware_audit — pin the auth middleware's real contract.

The middleware is the API's only auth surface, and its edges are subtle:
an *empty-string* ``QUANT_API_KEY`` behaves like an unset key (loopback
gate, not key auth); non-loopback clients are refused entirely when the
key is unset; the 64 KiB body cap is enforced twice — pre-emptively on
``Content-Length`` and again while streaming; security headers apply to
4xx responses too. Every one of those edges is a fail-closed decision —
this lane pins them so drift is loud. Sealed ``middleware_audit.v1``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi.testclient import TestClient

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["middleware_audit", "middleware_audit_bench"]


def _client() -> TestClient:
    from quant_fund.api.app import app

    return TestClient(app, raise_server_exceptions=False)


def _nonloopback_status(path: str, expected: str | None) -> int:
    """Drive ``_authenticate_request`` with a forged remote client scope."""
    from starlette.requests import Request

    from quant_fund.api.app import _authenticate_request

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "client": ("203.0.113.9", 51000),
        "query_string": b"",
        "scheme": "http",
        "server": ("example.test", 80),
        "root_path": "",
        "http_version": "1.1",
        "raw_path": path.encode(),
    }

    async def _run() -> int:
        request = Request(scope)

        async def call_next(_req: Request) -> Any:
            class _Resp:
                status_code = 200
                headers: dict[str, str] = {}

            return _Resp()

        resp = await _authenticate_request(request, call_next, path, expected)
        return int(resp.status_code)

    return asyncio.run(_run())


def middleware_audit() -> dict[str, Any]:
    results: dict[str, Any] = {}
    saved = os.environ.pop("QUANT_API_KEY", None)
    try:
        client = _client()
        r = client.get("/health")
        results["public_path_no_key"] = {"status": r.status_code, "ok": r.status_code == 200}

        results["loopback_no_key_allowed"] = {
            "status": client.get("/models").status_code,
        }

        results["remote_no_key_refused"] = {
            "status": _nonloopback_status("/models", None),
            "ok": _nonloopback_status("/models", None) == 403,
            "remote_on_public_path": _nonloopback_status("/health", None) == 200,
        }

        # Empty-string key must NOT silently enable key auth semantics —
        # pinned: it falls through to the loopback gate like unset.
        os.environ["QUANT_API_KEY"] = ""
        results["empty_key_equals_unset"] = {
            "remote_status": _nonloopback_status("/models", ""),
            "loopback_status": _client().get("/models").status_code,
        }

        os.environ["QUANT_API_KEY"] = "sekrit-test-key"
        c2 = _client()
        results["key_required"] = {
            "no_key_status": c2.get("/models").status_code,
            "wrong_key_status": c2.get("/models", headers={"X-API-Key": "wrong"}).status_code,
            "right_key_status": c2.get(
                "/models", headers={"X-API-Key": "sekrit-test-key"}
            ).status_code,
            "public_still_public": c2.get("/health").status_code,
        }
        os.environ.pop("QUANT_API_KEY", None)

        c3 = _client()
        results["body_limits"] = {
            "oversized_declared": c3.post(
                "/optimize",
                content=b"x" * (64 * 1024 + 1),
                headers={"Content-Type": "application/json"},
            ).status_code,
            "malformed_length": c3.post(
                "/optimize",
                content=b"{}",
                headers={"Content-Length": "notanint", "Content-Type": "application/json"},
            ).status_code,
        }

        # Headers must apply to error responses too — probe a real 401.
        os.environ["QUANT_API_KEY"] = "sekrit-test-key"
        r_err = _client().get("/models")
        os.environ.pop("QUANT_API_KEY", None)
        results["security_headers_on_errors"] = {
            "nosniff": r_err.headers.get("X-Content-Type-Options"),
            "frame": r_err.headers.get("X-Frame-Options"),
            "csp_present": "Content-Security-Policy" in r_err.headers,
            "cache": r_err.headers.get("Cache-Control"),
            "on_status": r_err.status_code,
        }
    finally:
        if saved is not None:
            os.environ["QUANT_API_KEY"] = saved
        else:
            os.environ.pop("QUANT_API_KEY", None)
    return results


def middleware_audit_bench() -> dict[str, Any]:
    r = middleware_audit()
    ok = (
        r["public_path_no_key"]["ok"]
        and r["loopback_no_key_allowed"]["status"] == 200
        and r["remote_no_key_refused"]["status"] == 403
        and r["remote_no_key_refused"]["remote_on_public_path"]
        and r["empty_key_equals_unset"]["remote_status"] == 403
        and r["empty_key_equals_unset"]["loopback_status"] == 200
        and r["key_required"]["no_key_status"] == 401
        and r["key_required"]["wrong_key_status"] == 401
        and r["key_required"]["right_key_status"] == 200
        and r["key_required"]["public_still_public"] == 200
        and r["body_limits"]["oversized_declared"] == 413
        and r["body_limits"]["malformed_length"] == 400
        and r["security_headers_on_errors"]["nosniff"] == "nosniff"
        and r["security_headers_on_errors"]["frame"] == "DENY"
        and r["security_headers_on_errors"]["csp_present"]
        and r["security_headers_on_errors"]["cache"] == "no-store"
        and r["security_headers_on_errors"]["on_status"] == 401
    )
    payload: dict[str, Any] = {
        "kind": "middleware_audit",
        "schema": "middleware_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Auth middleware contract pinned: public-path bypass, loopback "
            "gate when unset, remote refusal, empty-key treated as unset, "
            "401 on missing/wrong key, 400/413 body limits, security "
            "headers present even on error responses."
            if ok
            else f"MIDDLEWARE DRIFT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
