"""apisurface_audit — the HTTP surface's contract, pinned end to end.

The ~30 endpoint batteries prove behavior *inside* each route; this one
pins the surface itself — the shape requests meet before any handler
runs:

- *inventory* — every registered APIRoute: the required paths exist,
  no (method, path) pair is bound twice, and the admin-scope map
  (``/harness/keys*``, ``/harness/drain``) is enforced by the shared
  scope function rather than per-route memory.
- *public surface* — exactly ``/health`` and nothing else; health
  stays public whether or not an env key is armed.
- *auth wiring* — auth precedes routing (an unknown path 401s
  unauthenticated, 404s authenticated), ``Authorization: Bearer`` is
  honored only under ``/v1``, ``X-API-Key`` works everywhere, dev mode
  trusts only loopback, and minted keys inherit the scope ladder
  (read < write < admin) with 403 ``insufficient_scope`` in the path's
  own error grammar.
- *error grammar* — 404/405/422 each answer in the path family's
  dialect: OpenAI ``{"error": ...}`` under ``/v1``,
  ``{"detail","code"}`` elsewhere.
- *headers* — ``X-Request-ID`` minted or echoed,
  ``X-Content-Type-Options: nosniff``, ``X-Fx1-Api-Version`` always.
- *drain* — the latch is one-way: gated work 503s ``draining`` while
  liveness answers, and re-POSTing is idempotent.
- *CORS* — off by default, wildcard refused, an explicit origin gets
  unauthenticated preflight (correctly — preflights carry no
  credentials) with only that origin echoed back.

Everything is a literal bool; the bench seals a
``apisurface_audit.v1`` receipt.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["apisurface_audit", "apisurface_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ENV_KEY = "audit-surface-key"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_API_CORS_ORIGINS",
)

_P_HEALTH = "/health"
_P_READY = "/ready"
_P_MODELS = "/v1/models"
_P_CHAT = "/v1/chat/completions"
_P_COMPLETE = "/harness/complete"
_P_KEYS = "/harness/keys"
_P_DRAIN = "/harness/drain"
_P_COMMANDS = "/harness/commands"
_P_VERIFY = "/receipts/verify"
_P_UNKNOWN = "/v1/definitely-not-here"
_ORIGIN_EVIL = "https://evil.example"
_ORIGIN_OK = "https://ok.example"
_REMOTE_CLIENT = "8.8.8.8"  # NOSONAR(S1313) — fixed non-loopback probe host

# Every route family a client can legitimately expect — the surface
# contract this battery pins. Deliberately conservative: only paths
# whose existence is load-bearing across dialects.
_REQUIRED_ROUTES: dict[str, set[str]] = {
    _P_HEALTH: {"GET"},
    _P_READY: {"GET"},
    "/metrics": {"GET"},
    _P_MODELS: {"GET"},
    _P_CHAT: {"POST"},
    "/v1/completions": {"POST"},
    "/v1/messages": {"POST"},
    "/v1/responses": {"POST"},
    "/v1/evals": {"GET", "POST"},
    "/v1/fine_tuning/jobs": {"GET", "POST"},
    _P_COMPLETE: {"POST"},
    "/harness/jobs": {"GET", "POST"},
    _P_KEYS: {"GET", "POST"},
    _P_DRAIN: {"POST"},
    _P_COMMANDS: {"GET"},
    _P_VERIFY: {"POST"},
}


def _build(
    api_key: str | None = None,
    client: tuple[str, int] | None = None,
    **kwargs: Any,
) -> tuple[TestClient, Any]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        for k in (*_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY"):
            os.environ.pop(k, None)
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(harness=Harness(runner=fake_runner), **kwargs)
        tc_kwargs: dict[str, Any] = {"raise_server_exceptions": False}
        if client is not None:
            tc_kwargs["client"] = client
        return TestClient(app, **tc_kwargs), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _route_map(app: FastAPI) -> dict[str, set[str]]:
    from fastapi.routing import APIRoute

    out: dict[str, set[str]] = {}
    for r in app.routes:
        if isinstance(r, APIRoute):
            out.setdefault(r.path, set()).update(r.methods or set())
    return out


def _probe_inventory() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, api_mod = _build()
    app = cast("FastAPI", client.app)
    routes = _route_map(app)
    for path, methods in _REQUIRED_ROUTES.items():
        tag = path.strip("/").replace("/", "_").replace(".", "_") or "root"
        out[f"rt_{tag}"] = path in routes and methods <= routes[path]
    out["rt_count_floor"] = len(routes) >= 60  # the surface is deep; pin a floor
    # no (method, path) pair bound twice
    seen: set[tuple[str, str]] = set()
    dupes = 0
    from fastapi.routing import APIRoute

    for r in app.routes:
        if isinstance(r, APIRoute):
            for m in r.methods or set():
                if (m, r.path) in seen:
                    dupes += 1
                seen.add((m, r.path))
    out["rt_no_duplicate_bindings"] = dupes == 0
    # the scope map is shared and honest
    out["rt_scope_admin_keys"] = api_mod._required_scope("GET", _P_KEYS) == "admin"  # noqa: SLF001
    out["rt_scope_admin_keys_nested"] = (
        api_mod._required_scope("DELETE", "/harness/keys/abc") == "admin"  # noqa: SLF001
    )
    out["rt_scope_admin_drain"] = api_mod._required_scope("POST", _P_DRAIN) == "admin"  # noqa: SLF001
    out["rt_scope_get_read"] = api_mod._required_scope("GET", _P_MODELS) == "read"  # noqa: SLF001
    out["rt_scope_post_write"] = (
        api_mod._required_scope("POST", _P_CHAT) == "write"  # noqa: SLF001
    )
    out["rt_scope_options_read"] = api_mod._required_scope("OPTIONS", "/v1/x") == "read"  # noqa: SLF001
    out["rt_scope_delete_write"] = (
        api_mod._required_scope("DELETE", "/v1/evals/e1") == "write"  # noqa: SLF001
    )
    return out


def _probe_public() -> dict[str, bool]:
    out: dict[str, bool] = {}
    dev, api_mod = _build()
    out["pb_public_exactly_health"] = frozenset({_P_HEALTH}) == api_mod._PUBLIC_PATHS  # noqa: SLF001
    out["pb_health_dev"] = dev.get(_P_HEALTH).status_code == 200
    secured, _ = _build(_ENV_KEY)
    out["pb_health_key_mode"] = secured.get(_P_HEALTH).status_code == 200
    for path in (_P_MODELS, _P_COMMANDS, "/metrics", _P_READY, "/openapi.json"):
        tag = path.strip("/").replace("/", "_").replace(".", "_")
        out[f"pb_gated_{tag}"] = secured.get(path).status_code == 401
    return out


def _probe_auth_surface() -> dict[str, bool]:
    out: dict[str, bool] = {}
    secured, _ = _build(_ENV_KEY)
    root = {"X-API-Key": _ENV_KEY}
    out["au_wrong_key_401"] = (
        secured.get(_P_MODELS, headers={"X-API-Key": "wrong-key"}).status_code == 401
    )
    out["au_env_key_200"] = secured.get(_P_MODELS, headers=root).status_code == 200
    out["au_bearer_v1"] = (
        secured.get(_P_MODELS, headers={"Authorization": f"Bearer {_ENV_KEY}"}).status_code == 200
    )
    out["au_bearer_harness_refused"] = (
        secured.get(_P_COMMANDS, headers={"Authorization": f"Bearer {_ENV_KEY}"}).status_code == 401
    )
    out["au_xkey_harness"] = secured.get(_P_COMMANDS, headers=root).status_code == 200
    # auth precedes routing: unknown path 401s unauthenticated, 404s authed
    out["au_unknown_unauth_401"] = secured.get(_P_UNKNOWN).status_code == 401
    out["au_unknown_authed_404"] = secured.get(_P_UNKNOWN, headers=root).status_code == 404
    out["au_unknown_harness_404"] = (
        secured.get("/harness/definitely-not-here", headers=root).status_code == 404
    )
    # dev mode: loopback trusted — TestClient host is in _LOOPBACK_HOSTS
    dev, api_mod = _build()
    out["au_dev_loopback_200"] = dev.get(_P_MODELS).status_code == 200
    out["au_dev_testclient_loopback"] = "testclient" in api_mod._LOOPBACK_HOSTS  # noqa: SLF001
    # non-loopback host in dev mode is refused, not silently trusted
    remote_dev, _ = _build(client=(_REMOTE_CLIENT, 31337))
    out["au_dev_remote_403"] = remote_dev.get(_P_MODELS).status_code == 403
    return out


def _probe_scopes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    secured, _ = _build(_ENV_KEY)
    root = {"X-API-Key": _ENV_KEY}
    mint = secured.post(_P_KEYS, json={"name": "svc"}, headers=root)
    mkey = mint.json().get("key", "") if mint.status_code == 201 else ""
    mh = {"X-API-Key": mkey}
    out["sc_mint_201"] = mint.status_code == 201 and mkey.startswith("fx1k_")
    out["sc_read_ok"] = secured.get(_P_MODELS, headers=mh).status_code == 200
    out["sc_write_ok"] = (
        secured.post(_P_VERIFY, json={"receipt": {"a": 1}}, headers=mh).status_code != 403
    )
    out["sc_admin_denied"] = secured.get(_P_KEYS, headers=mh).status_code == 403
    denied = secured.get(_P_KEYS, headers=mh)
    out["sc_denied_code"] = denied.json().get("code") == "insufficient_scope"
    out["sc_admin_env_ok"] = secured.get(_P_KEYS, headers=root).status_code == 200
    out["sc_drain_admin_only"] = secured.post(_P_DRAIN, headers=mh).status_code == 403
    # read-only minted key cannot write
    mint_ro = secured.post(_P_KEYS, json={"name": "ro", "scopes": ["read"]}, headers=root)
    rokey = mint_ro.json().get("key", "") if mint_ro.status_code == 201 else ""
    roh = {"X-API-Key": rokey}
    out["sc_ro_read_ok"] = secured.get(_P_MODELS, headers=roh).status_code == 200
    ro_denied = secured.post(_P_VERIFY, json={"receipt": {"a": 1}}, headers=roh)
    out["sc_ro_write_403"] = (
        ro_denied.status_code == 403 and ro_denied.json().get("code") == "insufficient_scope"
    )
    # v1-scope denial uses the OpenAI error grammar
    v1_denied = secured.post("/v1/evals", json={}, headers=roh)
    out["sc_v1_denied_openai_shape"] = v1_denied.status_code == 403 and "error" in v1_denied.json()
    return out


def _probe_error_grammar() -> dict[str, bool]:
    out: dict[str, bool] = {}
    secured, _ = _build(_ENV_KEY)
    root = {"X-API-Key": _ENV_KEY}
    v1_404 = secured.get(_P_UNKNOWN, headers=root)
    out["eg_v1_404_openai"] = v1_404.status_code == 404 and "error" in v1_404.json()
    h_404 = secured.get("/harness/definitely-not-here", headers=root)
    out["eg_harness_404_detail"] = (
        h_404.status_code == 404 and "detail" in h_404.json() and "code" in h_404.json()
    )
    # harness surfaces 405 in the detail/code grammar; /v1 method
    # mismatches land in the OpenAI-parity catch-all's 404 Invalid URL
    out["eg_harness_method_405"] = (
        secured.get(_P_COMPLETE, headers=root).status_code == 405
        and secured.get(_P_COMPLETE, headers=root).json().get("code") == "method_not_allowed"
    )
    v1_mm = secured.request("PUT", _P_CHAT, headers=root)
    out["eg_v1_wrong_method_404"] = v1_mm.status_code == 404 and "error" in v1_mm.json()
    # GET /v1/chat/completions is a real route (chat-store list), not a miss
    out["eg_v1_list_route_real"] = secured.get(_P_CHAT, headers=root).status_code == 200
    v1_422 = secured.post(_P_CHAT, json={}, headers=root)
    out["eg_v1_422_openai"] = v1_422.status_code == 422 and "error" in v1_422.json()
    u_401 = secured.get(_P_MODELS)
    out["eg_v1_401_openai"] = u_401.status_code == 401 and "error" in u_401.json()
    h_401 = secured.get(_P_COMMANDS)
    out["eg_harness_401_detail"] = (
        h_401.status_code == 401 and "detail" in h_401.json() and "code" in h_401.json()
    )
    return out


def _probe_headers() -> dict[str, bool]:
    out: dict[str, bool] = {}
    dev, _ = _build()
    r = dev.get(_P_HEALTH)
    out["hd_request_id_stamped"] = bool(r.headers.get("x-request-id"))
    echoed = dev.get(_P_HEALTH, headers={"X-Request-ID": "audit-req-1"})
    out["hd_request_id_echo"] = echoed.headers.get("x-request-id") == "audit-req-1"
    out["hd_nosniff"] = r.headers.get("x-content-type-options") == "nosniff"
    out["hd_api_version"] = bool(r.headers.get("x-fx1-api-version"))
    # error responses carry the same headers
    err = dev.get(_P_UNKNOWN)
    out["hd_errors_too"] = bool(err.headers.get("x-request-id"))
    return out


def _probe_drain() -> dict[str, bool]:
    out: dict[str, bool] = {}
    secured, _ = _build(_ENV_KEY)
    root = {"X-API-Key": _ENV_KEY}
    d = secured.post(_P_DRAIN, headers=root)
    out["dr_latch_200"] = d.status_code == 200 and d.json().get("draining") is True
    refused = secured.post(_P_COMPLETE, json={"command": "selftest"}, headers=root)
    out["dr_gated_503"] = refused.status_code == 503
    out["dr_gated_code"] = refused.json().get("code") == "draining"
    out["dr_health_survives"] = secured.get(_P_HEALTH).status_code == 200
    out["dr_ready_survives"] = secured.get(_P_READY, headers=root).status_code in (200, 503)
    d2 = secured.post(_P_DRAIN, headers=root)
    out["dr_idempotent"] = d2.status_code == 200 and d2.json().get("draining") is True
    return out


def _probe_cors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    dev, api_mod = _build()
    preflight = dev.options(
        _P_CHAT,
        headers={
            "Origin": _ORIGIN_EVIL,
            "Access-Control-Request-Method": "POST",
        },
    )
    out["cors_default_closed"] = (
        "access-control-allow-origin" not in preflight.headers
        or preflight.headers["access-control-allow-origin"] != _ORIGIN_EVIL
    )
    out["cors_wildcard_refused"] = False
    try:
        api_mod.create_app(cors_origins="*")
    except ValueError:
        out["cors_wildcard_refused"] = True
    out["cors_bad_origin_refused"] = False
    try:
        api_mod.create_app(cors_origins="not-a-url")
    except ValueError:
        out["cors_bad_origin_refused"] = True
    ok_dev, _ = _build(cors_origins=_ORIGIN_OK)
    ok_pf = ok_dev.options(
        _P_CHAT,
        headers={
            "Origin": _ORIGIN_OK,
            "Access-Control-Request-Method": "POST",
        },
    )
    out["cors_explicit_origin"] = ok_pf.headers.get("access-control-allow-origin") == _ORIGIN_OK
    evil_pf = ok_dev.options(
        _P_CHAT,
        headers={
            "Origin": _ORIGIN_EVIL,
            "Access-Control-Request-Method": "POST",
        },
    )
    out["cors_unlisted_origin_denied"] = (
        evil_pf.headers.get("access-control-allow-origin") != _ORIGIN_EVIL
    )
    return out


def apisurface_audit() -> dict[str, bool]:
    """Every API-surface contract as booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_inventory())
    out.update(_probe_public())
    out.update(_probe_auth_surface())
    out.update(_probe_scopes())
    out.update(_probe_error_grammar())
    out.update(_probe_headers())
    out.update(_probe_drain())
    out.update(_probe_cors())
    return out


def apisurface_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the API-surface battery."""
    r = apisurface_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "apisurface_audit",
        "schema": "apisurface_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "TestClient through the full middleware stack",
            "not_verified": [
                "per-endpoint response semantics (each has a dedicated battery)",
                "TLS termination (handled at the proxy layer)",
            ],
        },
        "interpretation": (
            "API surface contract holds: the route inventory is pinned, "
            "auth precedes routing, scopes ladder read<write<admin in the "
            "path's own grammar, drain/liveness/CORS behave, and every "
            "response family speaks its dialect."
            if ok
            else f"APISURFACE AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(apisurface_audit_bench(), indent=2, sort_keys=True))
