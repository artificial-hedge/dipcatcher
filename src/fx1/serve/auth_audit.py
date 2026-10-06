"""auth_audit — adversarial probes on the authentication/authorization surface.

The claim under test: the wire admits exactly the credentials it declares
— the ``FX1_API_KEY`` root credential and minted ``fx1k_`` managed keys —
through exactly two channels (``X-API-Key`` on any path,
``Authorization: Bearer`` on ``/v1*``), in exactly one order
(authenticate, then authorize, then meter), and every refusal lands in
the path's own error envelope with a uniform body that leaks nothing
about which credentials exist. A credential that is wrong, malformed,
revoked, or expired is the same dead credential everywhere; a key whose
scopes don't cover the call gets 403 ``insufficient_scope``, not 401 —
and neither refusal moves a meter.

Coverage map:

- *header parsing* — ``X-API-Key`` (name case-insensitive) authenticates
  on every path; ``Authorization: Bearer`` is read only on the
  ``/v1*`` dialect prefix and only when ``X-API-Key`` is absent (or
  empty — an empty header is no header). The scheme match is exact:
  lowercase ``bearer``, a tab or doubled space, a missing space, or a
  non-Bearer scheme all refuse. Trailing/leading whitespace on the
  credential refuses. ``X-API-Key`` wins outright when both channels
  carry a credential — a garbage ``X-API-Key`` refuses even against a
  valid Bearer (no fallback), and a valid ``X-API-Key`` admits against
  a garbage Bearer. Duplicated header lines resolve first-wins on both
  channels. The env root key travels either channel on ``/v1``.
- *malformed credentials* — absent, no-prefix, prefix-only,
  prefix+garbage, wrong-case prefix, and almost-a-real-key all refuse
  the same uniform 401 — the body is byte-identical for absent vs
  wrong (no oracle for which entries exist). A credential carrying
  non-ASCII bytes is the same refusal, never a server fault. The same
  fail-closed rule covers a POSIX surrogate-escaped environment key
  (compare_digest is fed utf-8 bytes post-fix).
- *revocation timing* — a key revoked while a request is in-flight
  lets the admitted call complete; every later call 401s. A revoked
  admin key cannot mint, rotate, or patch. Rotating or patching a
  revoked *target* refuses 409 ``key_revoked``. The tombstone keeps
  its counters — the audit trail reports the dead key's lifetime
  spend. ``revoke`` itself is idempotent-refusing.
- *scope matrix* — literal membership, no hierarchy: ``read`` covers
  GET/HEAD/OPTIONS, ``write`` the mutating data-plane methods,
  ``admin`` the control plane (``/harness/keys*``, ``/harness/drain``)
  on *any* method — a read key cannot even GET the key list. A
  write-only key cannot read; an admin-only key cannot read or write.
  The admin boundary is slash-delimited: ``/harness/keys/x`` requires
  admin, ``/harness/keysXYZ`` does not. Method classification precedes
  routing: a write key's OPTIONS is refused 403 while a read key's
  OPTIONS reaches routing and 404s — and HEAD is never implicit (405
  on ``/health``).
- *auth-then-scope order* — a garbage key on an admin route 401s (the
  scope question never arises); a valid-but-under-scoped key on the
  same route 403s ``insufficient_scope``; a quota-exhausted key on the
  same route 429s — the budget check also precedes scope.
- *expiry* — a key past ``expires_at`` fails closed like a revoked one:
  uniform 401, on admin routes too (never a scope error). The boundary
  is exact at the store: ``t < expires_at`` admits, ``t == expires_at``
  refuses. Rotation inherits the absolute deadline — a dead
  predecessor's successor is born dead.
- *loopback / no-auth mode* — with neither ``FX1_API_KEY`` nor managed
  keys the loopback dev surface is fully trusted (bare requests admit,
  minting works without a credential, presented credentials are
  decorative); a non-loopback client 403s ``forbidden`` and cannot
  spoof ``request.client`` via forwarding headers. Provisioning the
  first key is itself what turns auth on — the next bare request 401s.
  With keys present, a remote caller with no credential 401s like
  everyone else; a missing ``request.client`` is treated as remote.
- *key material hygiene* — the raw secret is returned exactly once at
  mint (and once for a rotated successor); ``list``/``get``/``usage``/
  ``self``/rotate-record surfaces expose the ``key_id`` fingerprint and
  the 13-char ``prefix`` only — never the raw secret, never the
  sha256, never a ``_``-private counter. Refusal bodies never echo the
  presented credential.
- *mid-flight changes* — PATCHing a key's scopes mid-request lets the
  admitted write complete (auth resolved at entry); the next write
  403s while reads keep admitting.
- *metering interaction* — auth failures are pre-meter: a forged-key
  storm moves no counter and mints no record; a 403 scope refusal
  decrements neither ``uses`` nor the rpm window; a 401 response
  carries no budget headers (nothing to attribute). The ops surface
  itself is gated: ``/metrics``, ``/ready``, ``/openapi.json`` and
  ``/docs`` all demand a credential while ``/health`` stays public.
- *anthropic dialect auth* — ``x-api-key`` (the stock SDK's spelling,
  same header) plus ``anthropic-version`` authenticates on
  ``/v1/messages``; ``Authorization: Bearer`` works there too — the
  dialect and the credential channel are orthogonal, and the
  X-API-Key-wins precedence holds under the anthropic grammar.
- *CORS edge* — with origins configured, a true preflight
  (``Origin`` + ``Access-Control-Request-Method``) is answered by the
  CORS layer before auth (preflights carry no credentials — the
  documented contract), a disallowed origin 400s, a bare OPTIONS
  still authenticates, and an actual request with ``Origin`` still
  demands a credential.
- *error envelope* — every refusal lands in the path's own grammar:
  ``{detail, code}`` on ``/harness*``, OpenAI ``{error: {message,
  type, code}}`` on ``/v1*``, Anthropic ``{type: "error", error}`` on
  ``/v1/messages*`` — ``unauthorized`` vs ``insufficient_scope`` vs
  ``forbidden`` codes, ``authentication_error`` /
  ``permission_error`` types, ``X-Request-ID`` echoed, never a bare
  response. ``x-should-retry`` is absent on 401/403 — not a
  retry-mapped status.
- *client error map* — ``HarnessClient`` maps 401 and 403 alike to
  ``HarnessAuthError`` and serves a managed key end to end.
- *store units* — ``authenticate`` returns ``None`` for non-str,
  non-prefix, unknown-sha, revoked, and expired credentials without
  moving a counter; the scope check precedes every counter; the
  expiry boundary is exact.

The key store, journal replay, and wire middleware provide the behavior
under test. This is a stub-backed TestClient battery — loopback is
proven by ASGI-scope host (``testclient``), remote refusal by unit-level
``_resolve_auth`` calls on fabricated scopes — not live network,
process-crash, or SDK evidence.

Sealed ``auth_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient
    from starlette.requests import Request

    from fx1.serve.backends import SamplingParams

__all__ = ["auth_audit", "auth_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("auth_audit_resources")


@contextmanager
def _audit_context() -> Iterator[None]:
    """Restore ambient configuration and close all synthetic resources.

    Run this diagnostic in a dedicated process: its environment and
    rate-window overrides are process-wide, not application configuration.
    The lock serializes calls made through this module.
    """
    with _AUDIT_LOCK:
        saved = {
            name: value
            for name, value in os.environ.items()
            if name.startswith("FX1_") or name == "MOONSHOT_API_KEY"
        }
        for name in saved:
            os.environ.pop(name, None)
        try:
            with ExitStack() as resources:
                token = _RESOURCES.set(resources)
                try:
                    yield
                finally:
                    _RESOURCES.reset(token)
        finally:
            for name in list(os.environ):
                if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
                    os.environ.pop(name, None)
            os.environ.update(saved)


def _temporary_directory() -> Path:
    return Path(_RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="auth_audit_")))


_MODELS_PATH = "/v1/models"
_CHAT_PATH = "/v1/chat/completions"
_MESSAGES_PATH = "/v1/messages"
_KEYS_PATH = "/harness/keys"
_DRAIN_PATH = "/harness/drain"
_SELF_PATH = "/harness/self"
_JOBS_PATH = "/harness/jobs"
_HEALTH_PATH = "/health"
_READY_PATH = "/ready"
_METRICS_PATH = "/metrics"
_OPENAPI_PATH = "/openapi.json"
_DOCS_PATH = "/docs"

_H_KEY = "X-API-Key"
_H_KEY_LOWER = "x-api-key"
_H_AUTH = "Authorization"
_H_ANTH_VER = "anthropic-version"
_H_RETRY_AFTER = "retry-after"
_H_SHOULD_RETRY = "x-should-retry"
_H_REQUEST_ID = "x-request-id"
_H_ANTH_RID = "request-id"
_H_RL_LIMIT = "x-ratelimit-limit-requests"
_H_RL_REMAINING = "x-ratelimit-remaining-requests"
_H_RL_RESET = "x-ratelimit-reset-requests"
_H_BACKEND = "X-Fx1-Backend"
_H_ORIGIN = "Origin"
_H_ACRM = "Access-Control-Request-Method"

_CHAT_BODY = {"model": "byok", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES_BODY = {
    "model": "fx1",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}

# The public fields a key record may ever carry on the wire — anything
# else (the sha256, a raw secret, a ``_``-private window counter) is a
# leak.
_PUBLIC_RECORD_FIELDS = frozenset(
    {
        "id",
        "object",
        "name",
        "prefix",
        "admin",
        "scopes",
        "rpm",
        "max_requests",
        "max_tokens",
        "expires_at",
        "created_at",
        "enabled",
        "revoked_at",
        "uses",
        "tokens_used",
        "last_used_at",
        "rotated_from",
        "requests_remaining",
        "tokens_remaining",
        "window_remaining",
        "window_reset_s",
        "log_cap",
        "log_dropped",
        "served",
    }
)


class _OkBackend:
    """Instant stub — completes immediately with an echo."""

    def __init__(self, model: str = "stub-model") -> None:
        self._model = model
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172) — protocol signature
    ) -> str:
        self.calls += 1
        return f"ok:{messages[-1]['content']}"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _HoldBackend:
    """Blocks inside ``complete`` until released — parks a request
    in-flight so revocation/scope changes land mid-request."""

    def __init__(self, model: str = "stub-model") -> None:
        self._model = model
        self._gate = threading.Event()
        self.entered = threading.Event()

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.entered.set()
        self._gate.wait(timeout=10)
        return "ok"

    def release(self) -> None:
        self._gate.set()

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


def _client(
    backend: Any | None = None,
    api_key: str | None = _ROOT,
    *,
    state_dir: Path | None = None,
    cors_origins: str | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; a single
    shared stub backend instance so holds/enters stay observable."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: backend or _OkBackend(),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            cors_origins=cors_origins,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post(_KEYS_PATH, json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _key_card(client: TestClient, root_h: dict[str, str], key_id: str) -> dict[str, Any]:
    r = client.get(f"{_KEYS_PATH}/{key_id}/usage", headers=root_h)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _patch_key(key_id: str, root_h: dict[str, str], client: TestClient, **fields: Any) -> None:
    r = client.patch(f"{_KEYS_PATH}/{key_id}", json=fields, headers=root_h)
    assert r.status_code == 200, r.text


def _models(client: TestClient, headers: dict[str, str] | None = None) -> Any:
    return client.get(_MODELS_PATH, headers=headers or {})


def _chat(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(_CHAT_PATH, json=_CHAT_BODY, headers=headers)


def _messages(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _MESSAGES_PATH,
        json=_MESSAGES_BODY,
        headers={**headers, _H_BACKEND: "byok"},
    )


def _dup_get(client: TestClient, path: str, header_pairs: list[tuple[str, str]]) -> Any:
    """A request carrying duplicated header lines — refused or
    resolved, the policy is measured not assumed."""
    return client.request("GET", path, headers=header_pairs)


def _code(resp: Any) -> str | None:
    """The refusal code out of whichever envelope grammar the path
    speaks — ``{detail, code}``, OpenAI ``{error}``, or Anthropic
    ``{type: "error", error}``."""
    try:
        body = resp.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        if isinstance(body.get("code"), str):
            return str(body["code"])
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("code"), str):
            return str(err["code"])
    return None


def _request(
    headers: list[tuple[bytes, bytes]],
    *,
    method: str = "GET",
    path: str = "/v1/models",
    client: tuple[str, int] | None = ("10.9.9.9", 1234),
) -> Request:
    """A fabricated ASGI request for unit-level ``_resolve_auth`` calls —
    carries header bytes the HTTP layer itself decodes latin-1, plus a
    client host that no header can fake. Defaults to a remote host so
    the loopback arm must be constructed explicitly."""
    from starlette.requests import Request  # noqa: PLC0415

    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": client,
            "scheme": "http",
            "server": ("test", 80),
            "root_path": "",
        }
    )


def _resolve_auth_unit(
    api_mod: ModuleType, request: Request, api_key: str | None, store: Any
) -> Any:
    return api_mod._resolve_auth(request, api_key, store)


def _status_of(resolved: Any) -> int | str:
    from starlette.responses import JSONResponse  # noqa: PLC0415

    if isinstance(resolved, JSONResponse):
        return int(resolved.status_code)
    return "resolved"


def _all_refusals_enveloped(*resps: Any) -> bool:
    """Every refusal is parseable JSON — never a bare body."""
    for resp in resps:
        try:
            body = resp.json()
        except ValueError:
            return False
        if not isinstance(body, dict) or not body:
            return False
    return True


# ---------------------------------------------------------------------------
# header parsing — the two credential channels and their precedence
# ---------------------------------------------------------------------------


def _header_parsing_probes() -> dict[
    str, Any
]:  # NOSONAR(S3776) — scripted traffic fans out per edge
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}
    k_raw, _ = _mint(client, root_h)
    bearer = f"Bearer {k_raw}"

    # X-API-Key: any path, header-name case-insensitive
    out["x_api_key_admits"] = _models(client, {_H_KEY: k_raw}).status_code == 200
    out["x_api_key_name_lowercase"] = _models(client, {_H_KEY_LOWER: k_raw}).status_code == 200
    out["x_api_key_name_mixed"] = _models(client, {"X-Api-KeY": k_raw}).status_code == 200
    out["x_api_key_admits_off_v1"] = (
        client.get(_JOBS_PATH, headers={_H_KEY_LOWER: k_raw}).status_code == 200
    )

    # Authorization: Bearer — exact scheme, /v1* only
    out["bearer_admits_v1"] = _models(client, {_H_AUTH: bearer}).status_code == 200
    out["bearer_header_name_case_insensitive"] = (
        _models(client, {"authorization": bearer}).status_code == 200
    )
    out["bearer_scheme_lowercase_refused"] = (
        _models(client, {_H_AUTH: f"bearer {k_raw}"}).status_code == 401
    )
    out["bearer_scheme_uppercase_refused"] = (
        _models(client, {_H_AUTH: f"BEARER {k_raw}"}).status_code == 401
    )
    out["bearer_double_space_refused"] = (
        _models(client, {_H_AUTH: f"Bearer  {k_raw}"}).status_code == 401
    )
    out["bearer_trailing_space_refused"] = (
        _models(client, {_H_AUTH: f"{bearer} "}).status_code == 401
    )
    out["bearer_tab_separator_refused"] = (
        _models(client, {_H_AUTH: f"Bearer\t{k_raw}"}).status_code == 401
    )
    out["bearer_no_space_refused"] = _models(client, {_H_AUTH: f"Bearer{k_raw}"}).status_code == 401
    out["bearer_empty_value_refused"] = _models(client, {_H_AUTH: "Bearer "}).status_code == 401
    out["bearer_scheme_only_refused"] = _models(client, {_H_AUTH: "Bearer"}).status_code == 401
    out["non_bearer_scheme_ignored"] = (
        _models(client, {_H_AUTH: f"Basic {k_raw}"}).status_code == 401
    )
    out["bearer_ignored_off_v1"] = (
        client.get(_JOBS_PATH, headers={_H_AUTH: bearer}).status_code == 401
    )
    # the dialect prefix is /v1*, not exact paths — bearer is honored on
    # an unknown /v1 route (auth passes, routing 404s)
    out["bearer_honored_on_unknown_v1"] = (
        client.get("/v1/nonexistent-surface", headers={_H_AUTH: bearer}).status_code == 404
    )
    out["no_cred_on_unknown_v1_is_401_envelope"] = (
        client.get("/v1/nonexistent-surface").status_code == 401
        and _code(client.get("/v1/nonexistent-surface")) == "unauthorized"
    )

    # env root key rides the bearer channel on /v1 too — and stays
    # unmetered (no budget headers)
    env_resp = _models(client, {_H_AUTH: f"Bearer {_ROOT}"})
    out["env_key_bearer_v1_admits"] = env_resp.status_code == 200
    out["env_key_bearer_unmetered"] = _H_RL_LIMIT not in {k.lower() for k in env_resp.headers}

    # ambiguity is refused, never resolved: two authentication
    # headers on one request — in any value combination — land a 400
    # bad_request before credential resolution, so no precedence rule
    # can become a proxy/application interpretation gap.
    for name, headers in (
        ("x_api_key_empty_plus_bearer_400", {_H_KEY: "", _H_AUTH: bearer}),
        (
            "x_api_key_garbage_plus_bearer_400",
            {_H_KEY: "fx1k_garbage", _H_AUTH: bearer},
        ),
        (
            "x_api_key_valid_plus_bearer_400",
            {_H_KEY: k_raw, _H_AUTH: "Bearer fx1k_garbage"},
        ),
        ("env_key_plus_bearer_400", {_H_KEY: _ROOT, _H_AUTH: bearer}),
    ):
        r = _models(client, headers)
        out[name] = r.status_code == 400 and _code(r) == "bad_request"

    # whitespace on the credential itself is part of the compared bytes
    out["x_api_key_leading_ws_refused"] = _models(client, {_H_KEY: f" {k_raw}"}).status_code == 401
    out["x_api_key_trailing_ws_refused"] = _models(client, {_H_KEY: f"{k_raw} "}).status_code == 401

    # duplicated header lines — singleton headers refuse the
    # ambiguity outright (400 bad_request), both channels alike, order
    # irrelevant.
    for name, pairs in (
        (
            "dup_x_api_key_refused_400",
            [(_H_KEY_LOWER, k_raw), (_H_KEY_LOWER, "fx1k_bad")],
        ),
        (
            "dup_x_api_key_refused_order_independent_400",
            [(_H_KEY_LOWER, "fx1k_bad"), (_H_KEY_LOWER, k_raw)],
        ),
        (
            "dup_authorization_refused_400",
            [(_H_AUTH, f"Bearer {k_raw}"), (_H_AUTH, "Bearer fx1k_bad")],
        ),
    ):
        r = _dup_get(client, _MODELS_PATH, pairs)
        out[name] = r.status_code == 400 and _code(r) == "bad_request"
    return out


# ---------------------------------------------------------------------------
# malformed credentials — the uniform refusal
# ---------------------------------------------------------------------------


def _malformed_credential_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, api_mod = _client()
    root_h = {_H_KEY: _ROOT}
    k_raw, _ = _mint(client, root_h)

    absent = client.get(_JOBS_PATH)
    no_prefix = client.get(_JOBS_PATH, headers={_H_KEY: "deadbeef"})
    prefix_only = client.get(_JOBS_PATH, headers={_H_KEY: "fx1k_"})
    prefix_garbage = client.get(_JOBS_PATH, headers={_H_KEY: "fx1k_zzzz"})
    wrong_case = client.get(_JOBS_PATH, headers={_H_KEY: k_raw.upper()})
    almost = client.get(_JOBS_PATH, headers={_H_KEY: k_raw[:-1]})
    out["missing_header_401"] = absent.status_code == 401
    out["no_prefix_401"] = no_prefix.status_code == 401
    out["prefix_only_401"] = prefix_only.status_code == 401
    out["prefix_garbage_suffix_401"] = prefix_garbage.status_code == 401
    out["wrong_case_prefix_401"] = wrong_case.status_code == 401
    out["almost_valid_key_401"] = almost.status_code == 401

    # the refusal is uniform — absent vs wrong vs forged get the same
    # body (no oracle for which entries exist)
    out["refusal_uniform_harness_body"] = (
        absent.json() == no_prefix.json() == prefix_garbage.json()
        and absent.json()["code"] == "unauthorized"
    )
    v1_absent = _models(client)
    v1_wrong = _models(client, {_H_KEY: "fx1k_forged"})
    out["refusal_uniform_v1_body"] = (
        v1_absent.json() == v1_wrong.json() and _code(v1_absent) == "unauthorized"
    )
    out["refusal_never_bare"] = _all_refusals_enveloped(
        absent, no_prefix, prefix_only, prefix_garbage, v1_absent, v1_wrong
    )

    # a credential carrying non-ASCII bytes refuses like any other dead
    # credential — never a server fault (hmac.compare_digest refuses
    # non-ASCII str, so the comparison must receive encoded bytes).
    # POSIX may also surface undecodable environment bytes as surrogate-
    # escaped str; those must fail closed instead of escaping as a 500.
    from fx1.serve.keys import ApiKeyStore  # noqa: PLC0415

    store = ApiKeyStore()
    store.mint()
    nonascii_responses = [
        _resolve_auth_unit(api_mod, _request([(b"x-api-key", raw_hdr)]), _ROOT, store)
        for raw_hdr in (b"fx1k_\xff\xfe", b"fx1k_\xe2\x82\xac")
    ]
    malformed_env = _resolve_auth_unit(
        api_mod,
        _request([(b"x-api-key", b"fx1k_forged")]),
        "\udcff",
        store,
    )
    refusal_body = {
        "error": {
            "message": "invalid or missing API key",
            "type": "authentication_error",
            "param": None,
            "code": "unauthorized",
        }
    }
    out["nonascii_x_api_key_uniform_401"] = all(
        _status_of(response) == 401 and json.loads(response.body) == refusal_body
        for response in (*nonascii_responses, malformed_env)
    )
    res_bearer = _resolve_auth_unit(
        api_mod,
        _request([(b"authorization", b"Bearer fx1k_\xff")]),
        _ROOT,
        store,
    )
    out["nonascii_bearer_uniform_401"] = _status_of(res_bearer) == 401
    return out


# ---------------------------------------------------------------------------
# revocation timing — a dead credential dies everywhere, instantly
# ---------------------------------------------------------------------------


def _revocation_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    hold = _HoldBackend()
    client, _ = _client(backend=hold)
    root_h = {_H_KEY: _ROOT}

    # a request admitted before the revoke completes — auth resolved at
    # entry; the tombstone applies to the *next* call
    k_raw, k_id = _mint(client, root_h)
    k_h = {_H_KEY: k_raw}
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(_chat, client, k_h)
        assert hold.entered.wait(timeout=5), "request never reached the backend"
        d = client.delete(f"{_KEYS_PATH}/{k_id}", headers=root_h)
        out["revoke_while_inflight_succeeds"] = d.status_code == 200
        hold.release()
        resp = fut.result(timeout=5)
    out["revoked_midflight_admitted_completes"] = resp.status_code == 200
    out["revoked_next_call_401"] = _chat(client, k_h).status_code == 401
    out["revoked_next_read_401"] = _models(client, k_h).status_code == 401

    # the tombstone keeps the audit trail — counters frozen at the
    # spent values, still readable by the admin
    card = _key_card(client, root_h, k_id)
    out["revoked_record_reports_spend"] = (
        card["uses"] == 1 and card["enabled"] is False and card["revoked_at"] is not None
    )

    # a revoked admin key can mint/rotate/patch nothing
    a_raw, a_id = _mint(client, root_h, admin=True)
    a_h = {_H_KEY: a_raw}
    victim_raw, victim_id = _mint(client, root_h)
    assert client.delete(f"{_KEYS_PATH}/{a_id}", headers=root_h).status_code == 200
    out["revoked_admin_cannot_mint"] = (
        client.post(_KEYS_PATH, json={}, headers=a_h).status_code == 401
    )
    out["revoked_admin_cannot_rotate"] = (
        client.post(f"{_KEYS_PATH}/{victim_id}/rotate", json={}, headers=a_h).status_code == 401
    )
    out["revoked_admin_cannot_patch"] = (
        client.patch(f"{_KEYS_PATH}/{victim_id}", json={}, headers=a_h).status_code == 401
    )

    # a live admin key reaches all three verbs — auth admits, the verb
    # then answers on the target's own state
    live_raw, _live_id = _mint(client, root_h, admin=True)
    live_h = {_H_KEY: live_raw}
    out["live_admin_mints"] = client.post(_KEYS_PATH, json={}, headers=live_h).status_code == 201
    out["live_admin_patches"] = (
        client.patch(f"{_KEYS_PATH}/{victim_id}", json={"name": "v"}, headers=live_h).status_code
        == 200
    )

    # rotating or patching a revoked *target* refuses 409 key_revoked —
    # a dead secret cannot mint a live one
    assert client.delete(f"{_KEYS_PATH}/{k_id}", headers=root_h).status_code in (200, 409)
    out["rotate_dead_target_refused"] = (
        client.post(f"{_KEYS_PATH}/{k_id}/rotate", json={}, headers=root_h).status_code == 409
    )
    out["patch_dead_target_refused"] = (
        client.patch(f"{_KEYS_PATH}/{k_id}", json={}, headers=root_h).status_code == 409
    )
    out["revoke_idempotent_refuses"] = (
        client.delete(f"{_KEYS_PATH}/{k_id}", headers=root_h).status_code == 409
    )

    # rotation kills the predecessor secret in the same transaction
    rot = client.post(f"{_KEYS_PATH}/{victim_id}/rotate", json={}, headers=root_h)
    assert rot.status_code == 201, rot.text
    out["rotated_predecessor_dead"] = _models(client, {_H_KEY: victim_raw}).status_code == 401
    succ_raw = str(rot.json()["key"]["key"])
    out["rotated_successor_lives"] = _models(client, {_H_KEY: succ_raw}).status_code == 200
    return out


# ---------------------------------------------------------------------------
# scope matrix — literal membership, no hierarchy
# ---------------------------------------------------------------------------


def _scope_matrix_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}
    r_raw, _ = _mint(client, root_h, scopes=["read"])
    w_raw, _ = _mint(client, root_h, scopes=["write"])
    a_raw, _ = _mint(client, root_h, scopes=["admin"])
    d_raw, _ = _mint(client, root_h)  # default [read, write]
    r_h, w_h, a_h, d_h = (
        {_H_KEY: r_raw},
        {_H_KEY: w_raw},
        {_H_KEY: a_raw},
        {_H_KEY: d_raw},
    )

    # read scope: safe methods admit anywhere off the control plane
    out["read_get_models"] = _models(client, r_h).status_code == 200
    out["read_get_self"] = client.get(_SELF_PATH, headers=r_h).status_code == 200
    # method classification precedes routing — a read key's HEAD/OPTIONS
    # pass auth and refuse downstream (HEAD is never implicit: 405 on
    # /health, 404 on the /v1 catch-all)
    out["read_head_passes_auth"] = client.head(_MODELS_PATH, headers=r_h).status_code == 404
    out["head_never_implicit_on_health"] = client.head(_HEALTH_PATH).status_code == 405
    out["read_options_passes_auth"] = client.options(_MODELS_PATH, headers=r_h).status_code == 404
    # mutating methods refuse 403
    out["read_post_chat_denied"] = _chat(client, r_h).status_code == 403
    out["read_delete_denied"] = client.delete("/v1/files/f_missing", headers=r_h).status_code == 403
    out["read_patch_denied"] = client.patch("/v1/responses/x", headers=r_h).status_code == 403
    out["read_put_denied"] = client.put(_MODELS_PATH, headers=r_h).status_code == 403
    # the control plane wants admin on ANY method — GET included
    out["read_admin_get_denied"] = client.get(_KEYS_PATH, headers=r_h).status_code == 403
    out["read_admin_options_denied"] = client.options(_KEYS_PATH, headers=r_h).status_code == 403
    out["read_drain_denied"] = client.post(_DRAIN_PATH, headers=r_h).status_code == 403

    # write scope: no read hierarchy — a write-only key cannot read
    out["write_get_denied"] = _models(client, w_h).status_code == 403
    out["write_head_denied"] = client.head(_MODELS_PATH, headers=w_h).status_code == 403
    out["write_options_denied"] = client.options(_MODELS_PATH, headers=w_h).status_code == 403
    out["write_post_admits"] = _chat(client, w_h).status_code == 200
    out["write_admin_denied"] = client.post(_KEYS_PATH, json={}, headers=w_h).status_code == 403
    out["write_drain_denied"] = client.post(_DRAIN_PATH, headers=w_h).status_code == 403

    # admin scope: the control plane only — no data-plane rides along
    out["admin_only_mints"] = client.post(_KEYS_PATH, json={}, headers=a_h).status_code == 201
    out["admin_only_lists"] = client.get(_KEYS_PATH, headers=a_h).status_code == 200
    out["admin_only_cannot_read"] = _models(client, a_h).status_code == 403
    out["admin_only_cannot_write"] = _chat(client, a_h).status_code == 403
    # admin covers every method on its surface — a PUT reaches routing
    put_admin = client.put(_KEYS_PATH, headers=a_h)
    out["admin_put_reaches_routing"] = put_admin.status_code in (404, 405, 422)
    out["write_put_admin_denied"] = client.put(_KEYS_PATH, headers=w_h).status_code == 403

    # default scopes — the pre-scope contract [read, write]: data-plane
    # yes, control plane no
    out["default_read_admits"] = _models(client, d_h).status_code == 200
    out["default_write_admits"] = _chat(client, d_h).status_code == 200
    out["default_admin_denied"] = client.get(_KEYS_PATH, headers=d_h).status_code == 403

    # the admin boundary is slash-delimited, not a raw prefix
    out["keys_boundary_slash_delimited"] = (
        client.get("/harness/keysXYZ", headers=r_h).status_code == 404
        and client.get(f"{_KEYS_PATH}/x", headers=r_h).status_code == 403
    )
    out["keys_trailing_slash_admin"] = (
        client.get(f"{_KEYS_PATH}/", headers=r_h).status_code == 403
        and client.get(f"{_KEYS_PATH}/", headers=a_h).status_code == 200
    )
    # last: a real drain latches the process — everything after it on
    # this app would answer 503 draining, so the admit probe runs final
    out["admin_only_drains_ok"] = client.post(_DRAIN_PATH, headers=a_h).status_code == 200
    return out


# ---------------------------------------------------------------------------
# order of operations — authenticate, then authorize, then meter
# ---------------------------------------------------------------------------


def _ordering_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}
    r_raw, _ = _mint(client, root_h, scopes=["read"])
    r_h = {_H_KEY: r_raw}

    # bad credentials never reach the scope question — 401 not 403
    denied = client.post(_KEYS_PATH, json={}, headers={_H_KEY: "fx1k_forged"})
    out["garbage_admin_route_401_not_403"] = (
        denied.status_code == 401 and _code(denied) == "unauthorized"
    )
    absent = client.post(_KEYS_PATH, json={})
    out["absent_admin_route_401_not_403"] = absent.status_code == 401

    # a valid-but-under-scoped key on the same route — 403 insufficient_scope
    scoped = client.post(_KEYS_PATH, json={}, headers=r_h)
    out["under_scoped_admin_route_403"] = (
        scoped.status_code == 403 and _code(scoped) == "insufficient_scope"
    )

    # an expired key on the same route is 401 — a dead credential is a
    # dead credential, not a scope error
    e_raw, e_id = _mint(client, root_h, scopes=["read", "write", "admin"])
    _patch_key(e_id, root_h, client, expires_at=time.time() - 1)
    expired = client.post(_KEYS_PATH, json={}, headers={_H_KEY: e_raw})
    out["expired_admin_route_401_not_403"] = (
        expired.status_code == 401 and _code(expired) == "unauthorized"
    )

    # a quota-exhausted key refuses 429 before the scope question too
    q_raw, _ = _mint(client, root_h, max_requests=1)
    q_h = {_H_KEY: q_raw}
    assert _models(client, q_h).status_code == 200
    spent = _chat(client, q_h)
    out["quota_exhausted_not_scope_error"] = (
        spent.status_code == 429 and _code(spent) == "quota_exceeded"
    )
    spent_admin = client.post(_KEYS_PATH, json={}, headers=q_h)
    out["quota_exhausted_admin_429_not_403"] = (
        spent_admin.status_code == 429 and _code(spent_admin) == "quota_exceeded"
    )
    return out


# ---------------------------------------------------------------------------
# expiry — a dead credential fails closed like a revoked one
# ---------------------------------------------------------------------------


def _expiry_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}

    # minted ttl — the wire refuses once the second turns
    t_raw, _ = _mint(client, root_h, ttl_s=0.3)
    t_h = {_H_KEY: t_raw}
    admitted = _models(client, t_h).status_code
    time.sleep(0.4)
    expired = _models(client, t_h)
    out["ttl_expired_wire_401"] = admitted == 200 and expired.status_code == 401
    out["expired_is_unauthorized_code"] = _code(expired) == "unauthorized"

    # a PATCHed expiry takes effect on the next call — and on the
    # control plane too (401, not a scope denial)
    p_raw, p_id = _mint(client, root_h)
    p_h = {_H_KEY: p_raw}
    assert _models(client, p_h).status_code == 200
    _patch_key(p_id, root_h, client, expires_at=time.time() - 1)
    out["patched_expiry_refuses_next_call"] = _models(client, p_h).status_code == 401
    out["expired_cannot_admin"] = client.post(_KEYS_PATH, json={}, headers=p_h).status_code == 401

    # rotation inherits the absolute deadline — a dead predecessor's
    # successor is born dead (rotation never extends a lifetime)
    d_raw, d_id = _mint(client, root_h)
    _patch_key(d_id, root_h, client, expires_at=time.time() - 1)
    rot = client.post(f"{_KEYS_PATH}/{d_id}/rotate", json={}, headers=root_h)
    out["rotate_dead_predecessor_mints_dead_successor"] = (
        rot.status_code == 201
        and _models(client, {_H_KEY: rot.json()["key"]["key"]}).status_code == 401
    )
    out["rotate_dead_predecessor_itself_401"] = _models(client, {_H_KEY: d_raw}).status_code == 401

    # the boundary is exact at the store: t < expires_at admits,
    # t == expires_at refuses — no grace second
    from fx1.serve.keys import ApiKeyStore  # noqa: PLC0415

    t = [1000.0]
    store = ApiKeyStore(journal=None, clock=lambda: t[0])
    s_raw, _rec = store.mint(ttl_s=60)
    t[0] = 1059.999
    before = store.authenticate(s_raw)
    t[0] = 1060.0
    at_exact = store.authenticate(s_raw)
    t[0] = 1060.001
    after = store.authenticate(s_raw)
    out["expiry_boundary_exact"] = before is not None and at_exact is None and after is None
    return out


# ---------------------------------------------------------------------------
# loopback / no-auth mode — the dev surface and its boundary
# ---------------------------------------------------------------------------


def _loopback_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, api_mod = _client(api_key=None)
    from fx1.serve.keys import ApiKeyStore  # noqa: PLC0415

    # no env key, empty store: loopback is trusted end to end
    out["dev_mode_bare_loopback_admits"] = _models(client).status_code == 200
    out["dev_mode_ignores_credentials"] = (
        _models(client, {_H_KEY: "fx1k_garbage"}).status_code == 200
        and _models(client, {_H_AUTH: "Bearer fx1k_garbage"}).status_code == 200
    )
    out["dev_mode_bare_mint_provisions"] = client.post(_KEYS_PATH, json={}).status_code == 201
    # provisioning the first key is itself what turns auth on
    out["post_provision_bare_401"] = _models(client).status_code == 401
    out["post_provision_garbage_401"] = _models(client, {_H_KEY: "fx1k_garbage"}).status_code == 401

    # unit level: the loopback rule keys on request.client.host — no
    # forwarding header can fake it, and a missing client is remote
    empty_store = ApiKeyStore()
    remote = _resolve_auth_unit(api_mod, _request([]), None, empty_store)
    out["remote_noauth_403_forbidden"] = _status_of(remote) == 403
    import json as _json  # noqa: PLC0415

    remote_body = _json.loads(remote.body.decode())
    out["remote_noauth_code_forbidden"] = (
        remote_body.get("code") == "forbidden"
        or (remote_body.get("error") or {}).get("code") == "forbidden"
    )
    spoofed = _resolve_auth_unit(
        api_mod,
        _request([(b"x-forwarded-for", b"127.0.0.1"), (b"x-real-ip", b"127.0.0.1")]),
        None,
        empty_store,
    )
    out["remote_headers_cannot_spoof_loopback"] = _status_of(spoofed) == 403
    no_client = _resolve_auth_unit(api_mod, _request([], client=None), None, empty_store)
    out["none_client_treated_remote"] = _status_of(no_client) == 403
    remote_cred = _resolve_auth_unit(
        api_mod, _request([(b"x-api-key", b"fx1k_x")]), None, empty_store
    )
    out["remote_dev_mode_cred_still_403"] = _status_of(remote_cred) == 403
    loop = _resolve_auth_unit(api_mod, _request([], client=("127.0.0.1", 1234)), None, empty_store)
    out["loopback_dev_resolves_trusted"] = isinstance(loop, tuple) and loop == (None, True, None)

    # once keys exist the remote caller with no credential gets the
    # uniform 401 like everyone else — the 403 is dev-mode-only
    keyed_store = ApiKeyStore()
    keyed_store.mint()
    remote_keyed = _resolve_auth_unit(api_mod, _request([]), None, keyed_store)
    out["remote_with_keys_401"] = _status_of(remote_keyed) == 401
    remote_keyed_bad = _resolve_auth_unit(
        api_mod, _request([(b"x-api-key", b"fx1k_nope")]), None, keyed_store
    )
    out["remote_with_keys_bad_cred_401"] = _status_of(remote_keyed_bad) == 401
    return out


# ---------------------------------------------------------------------------
# key material hygiene — the secret appears exactly once
# ---------------------------------------------------------------------------


def _hygiene_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    import hashlib  # noqa: PLC0415

    client, _ = _client()
    root_h = {_H_KEY: _ROOT}

    mint = client.post(_KEYS_PATH, json={"rpm": 2}, headers=root_h)
    body = mint.json()
    raw = str(body["key"])
    sha = hashlib.sha256(raw.encode()).hexdigest()
    out["mint_returns_full_key_once"] = mint.status_code == 201 and body["key"] == raw
    out["mint_id_is_truncated_fingerprint"] = body["id"] == sha[:16] != raw
    out["mint_prefix_is_truncated"] = body["prefix"] == raw[:13] != raw

    key_h = {_H_KEY: raw}
    surfaces = {
        "list": client.get(_KEYS_PATH, headers=root_h).text,
        "get": client.get(f"{_KEYS_PATH}/{body['id']}", headers=root_h).text,
        "usage": client.get(f"{_KEYS_PATH}/{body['id']}/usage", headers=root_h).text,
        "self": client.get(_SELF_PATH, headers=key_h).text,
    }
    for tag, text in surfaces.items():
        out[f"{tag}_surface_never_serializes_raw"] = raw not in text
        out[f"{tag}_surface_never_serializes_sha256"] = sha not in text and '"sha256"' not in text
        out[f"{tag}_surface_never_serializes_private_fields"] = '"_' not in text
    # the mint body itself carries no sha256 and no _-private field
    out["mint_body_never_serializes_sha256"] = sha not in json.dumps(body)
    out["mint_body_never_serializes_private_fields"] = '"_' not in json.dumps(body)

    # every record field is declared public — no private counter or
    # hash field crosses the wire
    get_body = client.get(f"{_KEYS_PATH}/{body['id']}", headers=root_h).json()
    out["record_fields_only_public"] = set(get_body.keys()) <= _PUBLIC_RECORD_FIELDS

    # rotation: the successor secret appears once; the response's
    # predecessor lineage carries no secret
    rot = client.post(f"{_KEYS_PATH}/{body['id']}/rotate", json={}, headers=root_h)
    rot_body = rot.json()
    succ_raw = str(rot_body["key"]["key"])
    succ_sha = hashlib.sha256(succ_raw.encode()).hexdigest()
    out["rotate_returns_successor_once"] = (
        rot.status_code == 201 and rot_body["key"]["key"] == succ_raw
    )
    out["rotate_body_no_sha256"] = succ_sha not in json.dumps(rot_body)
    # ...and the successor is invisible everywhere after
    lst_after = client.get(_KEYS_PATH, headers=root_h).text
    out["successor_raw_never_resurfaces"] = succ_raw not in lst_after

    # refusal bodies never echo the presented credential
    forged = "fx1k_deadbeefcafe0001"
    bad = _models(client, {_H_KEY: forged})
    scoped_raw, _ = _mint(client, root_h, scopes=["read"])
    denied = _chat(client, {_H_KEY: scoped_raw})
    out["refusal_never_echoes_credential"] = forged not in bad.text
    out["scope_refusal_never_echoes"] = scoped_raw not in denied.text
    return out


# ---------------------------------------------------------------------------
# mid-flight policy changes — auth resolves at entry
# ---------------------------------------------------------------------------


def _midflight_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    hold = _HoldBackend()
    client, _ = _client(backend=hold)
    root_h = {_H_KEY: _ROOT}

    k_raw, k_id = _mint(client, root_h)
    k_h = {_H_KEY: k_raw}
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(_chat, client, k_h)
        assert hold.entered.wait(timeout=5), "request never reached the backend"
        pa = client.patch(f"{_KEYS_PATH}/{k_id}", json={"scopes": ["read"]}, headers=root_h)
        out["scope_patch_while_inflight_succeeds"] = pa.status_code == 200
        hold.release()
        resp = fut.result(timeout=5)
    out["inflight_write_completes_despite_patch"] = resp.status_code == 200
    out["patched_next_write_403"] = _chat(client, k_h).status_code == 403
    out["patched_read_still_admits"] = _models(client, k_h).status_code == 200
    return out


# ---------------------------------------------------------------------------
# metering — auth failures are pre-meter
# ---------------------------------------------------------------------------


def _metering_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}

    # a 403 scope refusal decrements neither uses nor the rpm window
    k_raw, k_id = _mint(client, root_h, scopes=["read"], rpm=5)
    k_h = {_H_KEY: k_raw}
    assert _models(client, k_h).status_code == 200  # uses=1, window 4/5
    denied = _chat(client, k_h)
    card = _key_card(client, root_h, k_id)
    out["scope_refusal_decrements_nothing"] = (
        denied.status_code == 403 and card["uses"] == 1 and card["window_remaining"] == 4
    )

    # a forged-credential storm mints no record and moves no meter —
    # the store's real keys stay at their spent values
    v_raw, v_id = _mint(client, root_h)
    v_h = {_H_KEY: v_raw}
    assert _models(client, v_h).status_code == 200  # uses=1
    n_keys = len(client.get(_KEYS_PATH, headers=root_h).json()["data"])
    for _ in range(5):
        _models(client, {_H_KEY: "fx1k_forged"})
    after = client.get(_KEYS_PATH, headers=root_h).json()["data"]
    out["forged_storm_mints_no_record"] = len(after) == n_keys
    out["forged_storm_moves_no_meter"] = _key_card(client, root_h, v_id)["uses"] == 1

    # a 401 response carries no budget headers — nothing to attribute
    refused = _models(client, {_H_KEY: "fx1k_forged"})
    h = {k.lower() for k in refused.headers}
    out["refusal_carries_no_budget_headers"] = (
        refused.status_code == 401
        and _H_RL_LIMIT not in h
        and _H_RL_REMAINING not in h
        and _H_RL_RESET not in h
    )

    # the ops surface itself is gated: only /health is public
    out["metrics_requires_auth"] = client.get(_METRICS_PATH).status_code == 401
    out["ready_requires_auth"] = client.get(_READY_PATH).status_code == 401
    out["openapi_requires_auth"] = client.get(_OPENAPI_PATH).status_code == 401
    out["docs_requires_auth"] = client.get(_DOCS_PATH).status_code == 401
    out["health_public_unmetered"] = client.get(_HEALTH_PATH).status_code == 200
    out["ops_surface_admits_with_key"] = (
        client.get(_METRICS_PATH, headers=v_h).status_code == 200
        and client.get(_OPENAPI_PATH, headers=v_h).status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# anthropic dialect auth — orthogonal channel, same credential
# ---------------------------------------------------------------------------


def _anthropic_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}
    k_raw, _ = _mint(client, root_h)

    # the stock SDK's spelling — x-api-key + anthropic-version — admits
    ok = _messages(client, {_H_KEY_LOWER: k_raw, _H_ANTH_VER: "2023-06-01"})
    out["anthropic_x_api_key_admits"] = ok.status_code == 200
    # the dialect is orthogonal to the credential channel — Bearer works
    # on /v1/messages too
    ok2 = _messages(client, {_H_AUTH: f"Bearer {k_raw}", _H_ANTH_VER: "2023-06-01"})
    out["anthropic_bearer_admits"] = ok2.status_code == 200
    # mixed channels under the anthropic grammar: refused ambiguous,
    # same as the OpenAI surface — never resolved by precedence.
    mixed = _messages(
        client,
        {_H_KEY: "fx1k_garbage", _H_AUTH: f"Bearer {k_raw}", _H_ANTH_VER: "2023-06-01"},
    )
    out["anthropic_mixed_refused_400"] = mixed.status_code == 400

    # anthropic grammar on refusals: {type: "error", error: {...}},
    # request-id echoed, x-should-retry absent (401/403 are not
    # retry-mapped statuses)
    denied = _messages(client, {_H_ANTH_VER: "2023-06-01"})
    denied_body = denied.json()
    out["anthropic_401_envelope"] = (
        denied.status_code == 401
        and denied_body.get("type") == "error"
        and isinstance(denied_body.get("error"), dict)
        and denied_body["error"].get("type") == "authentication_error"
    )
    out["anthropic_401_no_should_retry"] = _H_SHOULD_RETRY not in {
        k.lower() for k in denied.headers
    }
    out["anthropic_refusal_request_id"] = denied.headers.get(
        _H_ANTH_RID
    ) is not None and denied.headers.get(_H_ANTH_RID) == denied.headers.get(_H_REQUEST_ID)
    r_raw, _ = _mint(client, root_h, scopes=["read"])
    scoped = _messages(client, {_H_KEY_LOWER: r_raw, _H_ANTH_VER: "2023-06-01"})
    scoped_body = scoped.json()
    out["anthropic_403_envelope"] = (
        scoped.status_code == 403
        and scoped_body.get("type") == "error"
        and scoped_body["error"].get("type") == "permission_error"
    )
    out["anthropic_403_no_should_retry"] = _H_SHOULD_RETRY not in {
        k.lower() for k in scoped.headers
    }
    return out


# ---------------------------------------------------------------------------
# CORS edge — preflights bypass auth by design, actuals never do
# ---------------------------------------------------------------------------


def _cors_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client(cors_origins="http://example.test")

    preflight = client.options(
        _CHAT_PATH,
        headers={_H_ORIGIN: "http://example.test", _H_ACRM: "POST"},
    )
    out["cors_preflight_bypasses_auth"] = (
        preflight.status_code == 200
        and preflight.headers.get("access-control-allow-origin") == "http://example.test"
    )
    disallowed = client.options(
        _CHAT_PATH,
        headers={_H_ORIGIN: "http://evil.test", _H_ACRM: "POST"},
    )
    out["cors_disallowed_origin_refused"] = disallowed.status_code == 400
    out["plain_options_still_authenticates"] = client.options(_CHAT_PATH).status_code == 401
    out["cors_actual_request_still_authenticates"] = (
        _models(client, {_H_ORIGIN: "http://example.test"}).status_code == 401
    )
    return out


# ---------------------------------------------------------------------------
# error envelope — every refusal in the path's own grammar
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}
    r_raw, _ = _mint(client, root_h, scopes=["read"])

    h401 = client.get(_JOBS_PATH, headers={_H_KEY: "fx1k_nope"})
    h403 = client.post(_KEYS_PATH, json={}, headers={_H_KEY: r_raw})
    v401 = _models(client)
    v403 = _chat(client, {_H_KEY: r_raw})
    m401 = _messages(client, {})

    out["envelope_401_harness_shape"] = h401.status_code == 401 and h401.json() == {
        "detail": "invalid or missing X-API-Key",
        "code": "unauthorized",
    }
    out["envelope_403_harness_shape"] = (
        h403.status_code == 403
        and h403.json().get("code") == "insufficient_scope"
        and isinstance(h403.json().get("detail"), str)
    )
    v401b = v401.json().get("error", {})
    out["envelope_401_v1_shape"] = (
        v401.status_code == 401
        and v401b.get("code") == "unauthorized"
        and v401b.get("type") == "authentication_error"
        and isinstance(v401b.get("message"), str)
    )
    v403b = v403.json().get("error", {})
    out["envelope_403_v1_shape"] = (
        v403.status_code == 403
        and v403b.get("code") == "insufficient_scope"
        and v403b.get("type") == "permission_error"
    )
    out["envelope_401_anthropic_shape"] = (
        m401.status_code == 401 and m401.json().get("type") == "error"
    )
    out["refusals_parseable_json"] = _all_refusals_enveloped(h401, h403, v401, v403, m401)
    out["refusals_echo_request_id"] = all(
        r.headers.get(_H_REQUEST_ID) for r in (h401, h403, v401, v403)
    )
    out["supplied_request_id_echoed_on_refusal"] = (
        _models(client, {_H_REQUEST_ID: "audit-rid-1"}).headers.get(_H_REQUEST_ID) == "audit-rid-1"
    )
    return out


# ---------------------------------------------------------------------------
# client error map — the wire contract the SDK keys on
# ---------------------------------------------------------------------------


class _TransportSpy:
    """Counting transport: mirrors ``_urllib_transport``'s serialization
    verbatim and forwards into the TestClient."""

    def __init__(self, tc: TestClient) -> None:
        self._tc = tc
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,  # NOSONAR(S1172) — transport signature
    ) -> tuple[int, Mapping[str, str], bytes]:
        path = urllib.parse.urlparse(url).path
        if urllib.parse.urlparse(url).query:
            path += "?" + urllib.parse.urlparse(url).query
        self.calls.append((method, path))
        req_headers = {"Accept": "application/json", **headers}
        data: bytes | None
        if isinstance(payload, bytes):
            data = payload
        elif payload is not None:
            data = json.dumps(payload).encode()
            req_headers.setdefault("Content-Type", "application/json")
        else:
            data = None
        r = self._tc.request(method, path, content=data, headers=req_headers)
        return r.status_code, dict(r.headers), r.content


def _client_probes() -> dict[str, Any]:
    from fx1.serve.client import HarnessAuthError, HarnessClient  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client()
    root_h = {_H_KEY: _ROOT}

    # 401 maps to HarnessAuthError
    spy = _TransportSpy(client)
    hc = HarnessClient("http://auth-audit.dev", api_key="fx1k_forged", transport=spy)
    exc_401: HarnessAuthError | None = None
    try:
        hc.complete([{"role": "user", "content": "hi"}], backend="byok")
    except HarnessAuthError as exc:
        exc_401 = exc
    out["client_401_maps_auth_error"] = exc_401 is not None

    # 403 insufficient_scope maps to the same HarnessAuthError
    r_raw, _ = _mint(client, root_h, scopes=["read"])
    spy2 = _TransportSpy(client)
    hc2 = HarnessClient("http://auth-audit.dev", api_key=r_raw, transport=spy2)
    exc_403: HarnessAuthError | None = None
    try:
        hc2._json("GET", "/harness/keys")
    except HarnessAuthError as exc:
        exc_403 = exc
    out["client_403_maps_auth_error"] = exc_403 is not None

    # a managed key serves the client end to end
    k_raw, _ = _mint(client, root_h)
    spy3 = _TransportSpy(client)
    hc3 = HarnessClient("http://auth-audit.dev", api_key=k_raw, transport=spy3)
    resp = hc3.complete([{"role": "user", "content": "hi"}], backend="byok")
    out["client_managed_key_serves"] = str(resp.content).startswith("ok:")
    return out


# ---------------------------------------------------------------------------
# store unit probes — authenticate() edges beneath the wire
# ---------------------------------------------------------------------------


def _store_unit_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    from fx1.serve.keys import ApiKeyStore, KeyStoreError  # noqa: PLC0415

    store = ApiKeyStore(journal=None)
    raw, rec = store.mint(rpm=2, scopes=["read", "write"])
    key_id = str(rec["key_id"])

    not_a_str: Any = None
    out["store_none_raw_returns_none"] = store.authenticate(not_a_str) is None
    out["store_empty_raw_returns_none"] = store.authenticate("") is None
    out["store_non_prefix_returns_none"] = store.authenticate("deadbeef") is None
    out["store_unknown_sha_returns_none"] = store.authenticate("fx1k_" + "0" * 40) is None
    out["store_prefix_only_returns_none"] = store.authenticate("fx1k_") is None
    first = store.authenticate(raw)
    out["store_valid_admits_and_counts"] = first is not None and int(first["uses"]) == 1

    # the scope check precedes every counter — a denied call leaves
    # uses and the window untouched
    try:
        store.authenticate(raw, required_scope="admin")
        out["store_scope_denial_raises"] = False
    except KeyStoreError as exc:
        out["store_scope_denial_raises"] = exc.code == "insufficient_scope"
    got = store.get(key_id) or {}
    out["store_scope_denial_uses_frozen"] = int(got.get("uses", -1)) == 1
    ws = store.window_state(key_id)
    out["store_scope_denial_window_untouched"] = ws is not None and ws[:2] == (2, 1)

    # revocation: dead is dead immediately, uses frozen
    store.revoke(key_id)
    out["store_revoked_returns_none"] = store.authenticate(raw) is None
    out["store_revoked_uses_frozen"] = int((store.get(key_id) or {}).get("uses", -1)) == 1
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def auth_audit() -> dict[str, Any]:
    """Run the auth-surface battery; returns literal bools."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_header_parsing_probes())
        out.update(_malformed_credential_probes())
        out.update(_revocation_probes())
        out.update(_scope_matrix_probes())
        out.update(_ordering_probes())
        out.update(_expiry_probes())
        out.update(_loopback_probes())
        out.update(_hygiene_probes())
        out.update(_midflight_probes())
        out.update(_metering_probes())
        out.update(_anthropic_probes())
        out.update(_cors_probes())
        out.update(_envelope_probes())
        out.update(_client_probes())
        out.update(_store_unit_probes())
        return out


def auth_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under auth_audit.v1."""
    r = auth_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "auth_audit",
        "schema": "auth_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends; ASGI-scope fabrication for loopback/remote edges",
            "not_executed": [
                "TypeScript client runtime",
                "live network delivery (raw socket header bytes, TLS termination)",
                "process-crash/power-loss durability",
                "live provider billing",
            ],
        },
        "interpretation": (
            "The auth surface admits exactly its declared credentials "
            "through exactly two channels — X-API-Key on any path "
            "(header-name case-insensitive) and Authorization: Bearer on "
            "the /v1* dialect prefix only — with X-API-Key taking "
            "absolute precedence: an empty X-API-Key is no header and "
            "falls through to Bearer, while a garbage one refuses "
            "without consulting Bearer. The Bearer scheme match is "
            "exact (lowercase, doubled spaces, tabs, missing space, "
            "non-Bearer schemes all refuse), duplicated header lines "
            "resolve first-wins, and credential whitespace is part of "
            "the compared bytes. Every malformed, absent, revoked, or "
            "expired credential gets the same uniform 401 — byte-"
            "identical bodies, no oracle for which entries exist — "
            "including non-ASCII header bytes and POSIX surrogate-escaped "
            "environment values (compare_digest is fed utf-8 encodings "
            "so malformed material refuses instead of faulting). Scope "
            "is literal membership: read "
            "covers GET/HEAD/OPTIONS, write the mutating methods, admin "
            "the /harness/keys* + /harness/drain control plane on any "
            "method — no hierarchy (a write-only key cannot read, an "
            "admin-only key cannot touch the data plane), and the admin "
            "boundary is slash-delimited. Order is authenticate, "
            "authorize, meter: garbage keys 401 on admin routes, "
            "under-scoped keys 403 insufficient_scope, exhausted keys "
            "429 — and neither a 403 nor a forged-key storm moves a "
            "counter. Mid-request revocation and scope patches let the "
            "admitted call complete; the next call sees the new policy. "
            "Expiry fails closed at the exact second (t >= expires_at "
            "refuses; rotation inherits the deadline). Loopback dev "
            "mode is trusted only while the store is empty and keyed "
            "off request.client.host — unspoofable by headers; "
            "provisioning the first key is what turns auth on. The raw "
            "secret leaves the store exactly once at mint; every other "
            "surface exposes fingerprint and 13-char prefix only, and "
            "refusals never echo the credential. The Anthropic dialect "
            "authenticates x-api-key + anthropic-version or Bearer "
            "alike and refuses in its own error grammar; CORS "
            "preflights bypass auth by design while actual requests "
            "never do. HarnessClient maps 401/403 to HarnessAuthError."
            if ok
            else f"AUTH AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(auth_audit_bench(), indent=2, sort_keys=True))
