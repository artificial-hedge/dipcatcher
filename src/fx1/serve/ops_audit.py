"""ops_audit — adversarial probes on the operations surface itself.

The claim under test: the endpoints a load balancer, an orchestrator, and
an operator's dashboard poll — liveness, readiness, version, metrics,
capabilities, backend status, deep probes, docs, and the OpenAPI spec —
report the process's true state end to end, in the harness's own error
grammar, under the documented auth posture, through drain.

Coverage map:

- *Liveness* — ``GET /health`` is the one public path: 200 unauthed and
  under a wrong credential, honest ``backends`` presence-of-credentials
  flags (hosted_k3 needs ``MOONSHOT_API_KEY``; byok needs all three env
  vars; local_fx1 needs a checkpoint dir with ``modelcard.json`` plus a
  serve url/cmd), ``registered_commands`` counts the real registry, and
  the ``draining`` flag reports the latch without ever refusing.
- *Readiness* — ``GET /ready`` is keyed, reports the live gated-work
  ``inflight`` gauge, 503s ``draining`` once the latch is set, and is
  deliberately backend-blind: a process with zero configured backends is
  still ready to accept work.
- *Version* — ``GET /harness/version`` reports ``contract.API_VERSION``
  and ``fx1.__version__``, identical to ``/health.version``,
  ``/harness/capabilities.fx1_version``, and the OpenAPI ``info.version``
  — one source of truth, no hardcoded fallback.
- *Metrics* — ``/metrics`` JSON counts every response exactly once (the
  scrape observes the pre-scrape state), partitions by status class,
  ``uptime_s`` tracks the monotonic process clock, the inflight gauge
  and watermark reflect real gated work, and the Prometheus view selects
  on ``?format=prom`` or ``Accept: text/plain``/OpenMetrics.
- *Backend status + deep probe* — ``GET /harness/backends`` reports the
  circuit breaker's real state (open/cooldown/streak only after real
  faults) and caches each ``POST /harness/backends/{name}/probe``
  verdict; the probe resolves the real backend, returns a verdict
  (``ok:false`` + ``error_class``) rather than an HTTP fault, stays
  slot-gated (503 under drain), bypasses the breaker without feeding it,
  and meters under its own ``probe:<name>`` series.
- *Advisory preflights* — ``POST /harness/gate/check`` and
  ``/harness/score`` answer verdicts (never HTTP faults for gate
  refusals), never resolve a backend, and stay up under drain.
- *Docs + spec* — ``/docs``, ``/redoc``, and ``/openapi.json`` serve the
  developer surface; the spec is deterministic, declares the middleware's
  stamped headers, mirrors the route table's methods exactly (the
  ``/v1/{path:path}`` catch-all stays out), and — like every gated path —
  requires a credential when auth is armed. ``/`` is a clean enveloped
  404, not a fabricated index.
- *Drain interaction* — the whole read surface (health, ready's 503,
  metrics, version, capabilities, backends, openapi, docs) keeps
  answering while the latch is set; only the slot-gated probe refuses.
- *Auth posture* — ``/health`` is the only public path; every other ops
  endpoint 401s unauthed under an armed key (docs and the spec included),
  read scope covers GETs, drain is admin-only, and dev mode opens the
  loopback surface.
- *Header contract* — ``X-Request-ID`` echoes well-formed inbound ids and
  mints on absent/malformed; the security headers (``nosniff``,
  ``no-store``, ``no-referrer``, ``X-Fx1-Api-Version``) ride successes
  and refusals alike; ``openai-version`` and the Anthropic dialect
  headers (``request-id``, ``x-should-retry``, ``anthropic-ratelimit-*``)
  are scoped to ``/v1`` and never leak onto ops paths.
- *Content negotiation* — ``Accept: text/plain`` selects the Prometheus
  exposition only on ``/metrics``; every other ops endpoint keeps its
  honest JSON type instead of pretending to negotiate.
- *Envelope* — every ops refusal (401/403/404/405/422/429/503) lands in
  the ``{detail, code}`` envelope with no bare 5xx.

Two defects were found and fixed while building this battery:
``_is_anthropic_surface`` honored ``anthropic-version`` on *any* path, so
ops answers carried Anthropic-dialect headers (``request-id``,
``x-should-retry``, and — for rpm-windowed managed keys —
``anthropic-ratelimit-requests-*``), contradicting the documented
"/v1/messages tree plus /v1/* under the header" contract — the surface
check is now scoped to ``is_openai_path``; and the backend probe's
resolve-stage verdicts (``backend_unavailable``) returned through an
early ``_verdict`` that skipped ``record_complete``, so the documented
``probe:<name>`` verdict series silently missed resolver failures — a
monitoring scrape watching only the metric could never see them.

Sealed ``ops_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["ops_audit", "ops_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_ANTH_VER = {"anthropic-version": "2023-06-01"}

# Env the battery touches per-app: swept clean at entry, set explicitly for
# the configured-backend probes, restored at exit.
_SWEPT_ENVS = (
    _API_KEY_ENV,
    "MOONSHOT_API_KEY",
    "FX1_API_STATE_DIR",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_SDK_STATE_DIR",
    "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
)

# Gated read surface — every path here requires a credential once auth is
# armed; ``/health`` alone is public.
_OPS_GETS = (
    "/ready",
    "/metrics",
    "/harness/version",
    "/harness/capabilities",
    "/harness/backends",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
    "/",
)

# Ops operationIds the spec must carry.
_OPS_OPERATION_IDS = {
    "health",
    "ready",
    "get_metrics",
    "get_version",
    "get_capabilities",
    "backends_status",
    "backend_probe",
    "gate_check",
    "drain",
    "self_usage",
}

_COMMON_RESPONSE_HEADERS = (
    "x-request-id",
    "x-fx1-api-version",
    "x-content-type-options",
    "cache-control",
    "referrer-policy",
    "openai-processing-ms",
)

# Anthropic-dialect response headers — declared for /v1 only.
_ANTHROPIC_HEADER_NAMES = (
    "request-id",
    "x-should-retry",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
)

_RESOURCE_STACK: ContextVar[ExitStack | None] = ContextVar("ops_audit_resources", default=None)


class _StubBackend:
    """Deterministic completion stub: ok content + scripted usage."""

    def __init__(self, model: str, usage: dict[str, int] | None = None) -> None:
        self._model = model
        self._usage = dict(usage) if usage else None
        self.calls = 0
        self.last_usage: dict[str, int] | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.calls += 1
        self.last_usage = dict(self._usage) if self._usage is not None else None
        return f"ok:{messages[-1]['content']}"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _FaultBackend:
    """Provider fault: ``complete`` raises before any work is done."""

    def __init__(self, model: str, exc: Exception) -> None:
        self._model = model
        self._exc = exc
        self.calls = 0
        self.last_usage: dict[str, int] | None = None

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.calls += 1
        raise self._exc

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _ParkedBackend:
    """Gate backend: ``complete`` parks until released — deterministic
    inflight occupancy with no sleeps and no races."""

    def __init__(self, model: str) -> None:
        self._model = model
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=60.0)
        return f"ok:{messages[-1]['content']}"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


def _unconfigured(*a: Any, **k: Any) -> Any:
    """Resolver-side refusal: raises before any backend exists — the
    "no credentials configured" leg."""
    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

    raise BackendNotConfiguredError("no credentials configured")


@contextmanager
def _audit_env() -> Iterator[None]:
    """Sweep FX1_*/MOONSHOT_* ambient config for the battery and restore it
    on exit — env flags are process-wide, so probes that need configured
    backends set them inside their own app construction."""
    saved = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("FX1_") or name == "MOONSHOT_API_KEY"
    }
    for name in saved:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
                os.environ.pop(name, None)
        os.environ.update(saved)


@contextmanager
def _audit_resources() -> Iterator[None]:
    """Close every client, executor, and temp dir even when a probe raises."""
    with ExitStack() as stack:
        token = _RESOURCE_STACK.set(stack)
        try:
            yield
        finally:
            _RESOURCE_STACK.reset(token)


def _resources() -> ExitStack:
    stack = _RESOURCE_STACK.get()
    if stack is None:
        raise RuntimeError("audit app creation requires an audit resource context")
    return stack


def _test_client(app: Any) -> TestClient:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    stack = _resources()
    client = TestClient(app, raise_server_exceptions=False)
    stack.callback(client.close)
    return stack.enter_context(client)


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


def _make_app(
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = _ROOT,
    record_calls: list[Any] | None = None,
    **create_kw: Any,
) -> Any:
    """create_app under an isolated env; ``backend_map[name]`` are zero-arg
    factories (a factory may raise to model an unconfigured link).
    ``record_calls`` collects resolver invocations."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {"byok": lambda: _StubBackend("m-0")}

    def fake_resolve(name: str, *a: Any, **k: Any) -> Any:
        if record_calls is not None:
            record_calls.append((name, a, k))
        return backends[name]()

    resources = _resources()
    swept = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=_fast_runner),
            backend_resolver=fake_resolve,
            **create_kw,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return app
    finally:
        for k, sv in swept.items():
            if sv is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = sv


def _client(
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = _ROOT,
    **kw: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — the battery's standard wired app."""
    import fx1.serve.api as api_mod  # noqa: PLC0415

    app = _make_app(backend_map, api_key, **kw)
    return _test_client(app), api_mod


def _root_h() -> dict[str, str]:
    return {"X-API-Key": _ROOT}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=_root_h())
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _checkpoint_env() -> dict[str, str]:
    """A real local_fx1-configured env: a checkpoint dir carrying
    ``modelcard.json`` plus a serve command — what ``_backend_configured``
    actually reads."""
    ck = _resources().enter_context(tempfile.TemporaryDirectory(prefix="ops_audit_ckpt_"))
    (Path(ck) / "modelcard.json").write_text("{}")
    return {"FX1_CHECKPOINT_DIR": ck, "FX1_LOCAL_SERVE_CMD": "echo serve"}


def _enveloped(resp: Any) -> bool:
    """The harness wire-code contract off /v1: a JSON body with a string
    ``code`` plus ``detail``. The drain-503 refusal is enveloped too, so
    no status range is assumed here — callers pin the status separately."""
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and isinstance(body.get("code"), str) and "detail" in body


@contextmanager
def _env_set(env: dict[str, str]) -> Iterator[None]:
    """Set env vars for one probe leg and restore them on exit — flags
    like ``_backend_configured`` read the process env per request, so
    the var must be live while the client answers, not just at create."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _common_headers(resp: Any) -> bool:
    """The middleware's stamped headers on any response, success or not."""
    return all(resp.headers.get(h) is not None for h in _COMMON_RESPONSE_HEADERS)


def _anthropic_headers_absent(resp: Any) -> bool:
    """No Anthropic-dialect header rides a non-/v1 response."""
    return all(resp.headers.get(h) is None for h in _ANTHROPIC_HEADER_NAMES)


def _liveness_probes() -> dict[str, Any]:  # NOSONAR(S3776) — scripted env matrix + drain leg
    """``/health`` — the public liveness contract."""
    import fx1  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    r = client.get("/health")
    body = r.json()
    out["health_200_public"] = r.status_code == 200  # no credential at all
    out["health_public_wrong_key"] = (
        client.get("/health", headers={"X-API-Key": "fx1k_wrong"}).status_code == 200
    )
    out["health_shape"] = (
        body["status"] == "ok"
        and body["service"] == "fx1-harness-api"
        and isinstance(body["registered_commands"], int)
        and body["registered_commands"] > 0
        and body["draining"] is False
        and set(body["backends"]) == {"hosted_k3", "byok", "local_fx1"}
        and all(v is False for v in body["backends"].values())
    )
    out["health_version_is_package"] = body["version"] == fx1.__version__
    # the env credential is server-side material — never echoed in-band
    out["health_never_echoes_credentials"] = _ROOT not in r.text
    out["health_methods_refuse_405"] = all(
        resp.status_code == 405 and _enveloped(resp)
        for m in ("POST", "PUT", "DELETE")
        for resp in (client.request(m, "/health"),)
    )
    head = client.head("/health")
    out["health_head_405_empty_body"] = head.status_code == 405 and head.content == b""

    # presence-of-credentials flags are read off the live env at request
    # time — set them around the actual requests
    with _env_set(
        {
            "MOONSHOT_API_KEY": "mk",
            "FX1_BYOK_BASE_URL": "http://127.0.0.1:9/v1",
            "FX1_BYOK_API_KEY": "bk",
        }
    ):
        out["health_partial_byok_flag_false"] = client.get("/health").json()["backends"] == {
            "hosted_k3": True,
            "byok": False,
            "local_fx1": False,
        }
    with _env_set(
        {
            "MOONSHOT_API_KEY": "mk",
            "FX1_BYOK_BASE_URL": "http://127.0.0.1:9/v1",
            "FX1_BYOK_API_KEY": "bk",
            "FX1_BYOK_MODEL": "bm",
            **_checkpoint_env(),
        }
    ):
        out["health_full_env_flags_true"] = client.get("/health").json()["backends"] == {
            "hosted_k3": True,
            "byok": True,
            "local_fx1": True,
        }
    # liveness survives drain and reports the latch in-band
    client.post("/harness/drain", headers=h)
    rd = client.get("/health")
    out["health_survives_drain_reports_latch"] = (
        rd.status_code == 200 and rd.json()["draining"] is True
    )
    return out


def _readiness_probes() -> dict[str, Any]:
    """``/ready`` — readiness is accepting-work truth, not backend truth."""
    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    rr = client.get("/ready", headers=h)
    out["ready_200_shape"] = (
        rr.status_code == 200 and rr.json()["ready"] is True and rr.json()["inflight"] == 0
    )
    unauthed = client.get("/ready")
    out["ready_requires_credential"] = (
        unauthed.status_code == 401
        and _enveloped(unauthed)
        and unauthed.json()["code"] == "unauthorized"
    )
    out["ready_wrong_key_401"] = (
        client.get("/ready", headers={"X-API-Key": "fx1k_wrong"}).status_code == 401
    )
    # deliberately backend-blind: zero configured backends is still ready —
    # readiness is "accepting work", never "can serve a model"
    out["ready_backend_blind"] = client.get("/ready", headers=h).status_code == 200
    out["ready_methods_refuse_405"] = all(
        client.request(m, "/ready", headers=h).status_code == 405 for m in ("POST", "PUT", "DELETE")
    )
    out["ready_head_405"] = client.head("/ready", headers=h).status_code == 405

    # the inflight field reports real gated occupancy
    parked = _ParkedBackend("m-park")
    gate_client, _ = _client({"byok": lambda: parked})
    gh = _root_h()
    done = threading.Event()

    def _call() -> None:
        gate_client.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
            headers=gh,
        )
        done.set()

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    assert parked.entered.wait(10.0), "parked call never entered"
    try:
        mid = gate_client.get("/ready", headers=gh)
        out["ready_reflects_inflight"] = mid.status_code == 200 and mid.json()["inflight"] == 1
        # drain while occupied: readiness flips 503 enveloped, headers intact
        d = gate_client.post("/harness/drain", headers=gh)
        dr = gate_client.get("/ready", headers=gh)
        out["ready_503_draining_enveloped"] = (
            d.status_code == 200
            and d.json()["draining"] is True
            and d.json()["inflight"] == 1
            and dr.status_code == 503
            and dr.json()["code"] == "draining"
            and _common_headers(dr)
        )
        m = gate_client.get("/metrics", headers=gh).json()
        out["ready_inflight_visible_in_metrics"] = m["inflight"] == 1
    finally:
        parked.release.set()
        t.join(10.0)
    assert done.wait(10.0), "parked call never released"
    post = gate_client.get("/metrics", headers=gh).json()
    out["inflight_returns_zero_watermark_keeps_peak"] = (
        post["inflight"] == 0 and post["inflight_watermark"] >= 1
    )
    return out


def _version_probes() -> dict[str, Any]:
    """``/harness/version`` — one source of truth across every surface."""
    import fx1  # noqa: PLC0415
    import fx1.serve.api as api_mod  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()
    r = client.get("/harness/version", headers=h)
    body = r.json()
    out["version_200_shape"] = (
        r.status_code == 200
        and body["api_version"] == api_mod.API_VERSION
        and body["fx1_version"] == fx1.__version__
    )
    # the package's real release — never a placeholder constant
    out["version_not_hardcoded_placeholder"] = body["fx1_version"] == fx1.__version__ and body[
        "fx1_version"
    ] not in {"", "0.0.1", "dev"}
    health_v = client.get("/health").json()["version"]
    cap = client.get("/harness/capabilities", headers=h).json()
    spec_v = client.get("/openapi.json", headers=h).json()["info"]["version"]
    out["version_consistent_across_surfaces"] = (
        health_v == cap["fx1_version"] == spec_v == body["fx1_version"] == fx1.__version__
    )
    out["api_version_consistent"] = cap["api_version"] == api_mod.API_VERSION
    out["version_api_header_matches_body"] = (
        r.headers.get("x-fx1-api-version") == body["api_version"]
    )
    out["version_requires_credential"] = (
        client.get("/harness/version").status_code == 401
        and client.get("/harness/version", headers={"X-API-Key": "fx1k_wrong"}).status_code == 401
    )
    out["version_methods_refuse_405"] = all(
        resp.status_code == 405 and _enveloped(resp)
        for m in ("POST", "PUT", "DELETE")
        for resp in (client.request(m, "/harness/version", headers=h),)
    )
    head = client.head("/harness/version", headers=h)
    out["version_head_405_empty_body"] = head.status_code == 405 and head.content == b""
    client.post("/harness/drain", headers=h)
    rd = client.get("/harness/version", headers=h)
    out["version_survives_drain"] = rd.status_code == 200 and rd.json() == body
    return out


def _capabilities_probes() -> dict[str, Any]:
    """``/harness/capabilities`` — the discovery payload reports the real
    constructor config, never a hardcoded feature list."""
    import fx1  # noqa: PLC0415
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.evals import EVAL_SUITES  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()
    r = client.get("/harness/capabilities", headers=h)
    body = r.json()
    exp_roles = sorted({str(c.role) for c in Harness(runner=_fast_runner).list_commands()})
    out["capabilities_200_shape"] = (
        r.status_code == 200
        and body["api_version"] == api_mod.API_VERSION
        and body["fx1_version"] == fx1.__version__
        and body["eval_suites"] == list(EVAL_SUITES)
        and body["roles"] == exp_roles
        and isinstance(body["features"], dict)
        and isinstance(body["limits"], dict)
        and isinstance(body["backends"], dict)
    )
    out["capabilities_backends_match_health"] = (
        body["backends"] == client.get("/health").json()["backends"]
    )
    out["capabilities_requires_credential"] = client.get("/harness/capabilities").status_code == 401
    out["capabilities_methods_refuse_405"] = all(
        client.request(m, "/harness/capabilities", headers=h).status_code == 405
        for m in ("POST", "DELETE")
    )

    # constructor config is reported verbatim — a differently-configured
    # process advertises different limits and feature flags
    td = _resources().enter_context(tempfile.TemporaryDirectory(prefix="ops_audit_rcpts_"))
    custom = _client(
        max_inflight=4,
        rate_limit_rps=50.0,
        batch_max=7,
        store_max=9,
        cors_origins="https://example.com",
        breaker_threshold=0,
        byok_override=False,
        receipts_dir=td,
    )[0]
    cap2 = custom.get("/harness/capabilities", headers=h).json()
    out["capabilities_limits_reflect_ctor"] = (
        cap2["limits"]["max_inflight"] == 4.0
        and cap2["limits"]["rate_limit_rps"] == 50.0
        and cap2["limits"]["batch_max"] == 7.0
        and cap2["limits"]["store_max"] == 9.0
    )
    out["capabilities_features_reflect_config"] = (
        cap2["features"]["cors"] is True
        and cap2["features"]["breaker"] is False
        and cap2["features"]["byok_override"] is False
        and cap2["features"]["receipts_store"] is True
    )
    missing = _client(receipts_dir="/nonexistent-ops-audit-dir")[0]
    cap3 = missing.get("/harness/capabilities", headers=h).json()
    out["capabilities_receipts_store_false_when_absent"] = (
        cap3["features"]["receipts_store"] is False
        and cap3["features"]["cors"] is False
        and cap3["features"]["breaker"] is True
    )
    out["capabilities_all_limits_finite_numbers"] = all(
        isinstance(v, (int, float)) and v >= 0 for v in body["limits"].values()
    )
    client.post("/harness/drain", headers=h)
    out["capabilities_survives_drain"] = (
        client.get("/harness/capabilities", headers=h).status_code == 200
    )
    return out


def _backends_status_probes() -> dict[str, Any]:
    """``/harness/backends`` — config flags, circuit state, probe cache."""
    out: dict[str, Any] = {}
    client, _ = _client(breaker_threshold=2, breaker_cooldown_s=60.0)
    h = _root_h()
    r = client.get("/harness/backends", headers=h)
    body = r.json()
    out["backends_status_200_shape"] = (
        r.status_code == 200
        and set(body) == {"hosted_k3", "local_fx1", "byok"}
        and all(
            isinstance(e["configured"], bool)
            and e["circuit_open"] is False
            and e["consecutive_failures"] == 0
            and e["last_probe"] is None
            for e in body.values()
        )
    )
    out["backends_status_requires_credential"] = client.get("/harness/backends").status_code == 401
    out["backends_status_methods_405"] = (
        client.post("/harness/backends", headers=h).status_code == 405
    )

    # circuit honesty: a dying backend trips the breaker for real
    breaker_client, _ = _client(
        {
            "hosted_k3": lambda: _FaultBackend("m-dead", RuntimeError("provider down")),
            "byok": lambda: _StubBackend("m-0"),
        },
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    for _ in range(2):
        resp = breaker_client.post(
            "/harness/complete",
            json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
            headers=h,
        )
        # RuntimeError faults surface as 502 and still feed the breaker
        assert resp.status_code == 502 and resp.json()["code"] == "backend_failure", resp.text
    st = breaker_client.get("/harness/backends", headers=h).json()
    out["circuit_state_honest_after_faults"] = (
        st["hosted_k3"]["circuit_open"] is True
        and st["hosted_k3"]["consecutive_failures"] == 2
        and st["hosted_k3"]["cooldown_remaining_s"] > 0
        and st["byok"]["circuit_open"] is False  # a peer's faults never leak
    )
    # the next call fast-fails at the breaker — no new provider hit
    fast = breaker_client.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
        headers=h,
    )
    out["circuit_open_fails_fast"] = fast.status_code == 503
    # breaker disabled: faults never open a circuit
    nobr = _client(
        {"hosted_k3": lambda: _FaultBackend("m-dead", RuntimeError("provider down"))},
        breaker_threshold=0,
    )[0]
    for _ in range(3):
        nobr.post(
            "/harness/complete",
            json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
            headers=h,
        )
    nb = nobr.get("/harness/backends", headers=h).json()["hosted_k3"]
    out["breaker_disabled_never_opens"] = (
        nb["circuit_open"] is False and nb["consecutive_failures"] == 0
    )
    # configured flags reflect the live env across this surface too
    with _env_set(
        {
            "MOONSHOT_API_KEY": "mk",
            "FX1_BYOK_BASE_URL": "http://x",
            "FX1_BYOK_API_KEY": "bk",
            "FX1_BYOK_MODEL": "bm",
        }
    ):
        fb = client.get("/harness/backends", headers=h).json()
        out["backends_configured_matches_env"] = (
            fb["hosted_k3"]["configured"] is True
            and fb["byok"]["configured"] is True
            and fb["local_fx1"]["configured"] is False
        )
    client.post("/harness/drain", headers=h)
    out["backends_status_survives_drain"] = (
        client.get("/harness/backends", headers=h).status_code == 200
    )
    return out


def _backend_probe_probes() -> dict[str, Any]:  # NOSONAR(S3776) — verdict matrix over stub kinds
    """``POST /harness/backends/{name}/probe`` — deep health is a verdict,
    never an HTTP fault, and never feeds the breaker it bypasses."""
    out: dict[str, Any] = {}
    calls: list[Any] = []
    client, _ = _client(
        {
            "byok": lambda: _StubBackend("m-0", {"total_tokens": 3}),
            "hosted_k3": _unconfigured,
            "local_fx1": lambda: _FaultBackend("m-f", RuntimeError("provider down")),
        },
        record_calls=calls,
    )
    h = _root_h()

    ok = client.post("/harness/backends/byok/probe", headers=h)
    okb = ok.json()
    out["probe_ok_verdict"] = (
        ok.status_code == 200
        and okb["backend"] == "byok"
        and okb["ok"] is True
        and okb["model"] == "m-0"
        and okb["latency_ms"] >= 0
        and okb["error"] is None
        and okb["error_class"] is None
    )
    unconf = client.post("/harness/backends/hosted_k3/probe", headers=h)
    out["probe_unconfigured_verdict_not_fault"] = (
        unconf.status_code == 200
        and unconf.json()["ok"] is False
        and unconf.json()["error_class"] == "backend_unavailable"
        and unconf.json()["model"] is None
    )
    # local_fx1's checkpoint_dir is validated by the resolver wrapper: a
    # missing one is a client 422, not a verdict — and the kwarg is
    # refused on every other backend
    ck = _resources().enter_context(tempfile.TemporaryDirectory(prefix="ops_audit_ck_"))
    out["probe_local_fx1_requires_checkpoint_422"] = (
        _enveloped(client.post("/harness/backends/local_fx1/probe", headers=h))
        and client.post("/harness/backends/local_fx1/probe", headers=h).status_code == 422
    )
    out["probe_checkpoint_refused_on_nonlocal_422"] = (
        client.post(
            "/harness/backends/byok/probe", json={"checkpoint_dir": ck}, headers=h
        ).status_code
        == 422
    )
    # a resolved backend that faults at call time reports the exception
    # class as a verdict — never an HTTP fault
    fault = client.post("/harness/backends/local_fx1/probe", json={"checkpoint_dir": ck}, headers=h)
    out["probe_fault_reports_exc_class"] = (
        fault.status_code == 200
        and fault.json()["ok"] is False
        and fault.json()["error_class"] == "RuntimeError"
        and "provider down" in fault.json()["error"]
    )
    unknown = client.post("/harness/backends/nope/probe", headers=h)
    out["probe_unknown_name_422"] = unknown.status_code == 422 and _enveloped(unknown)
    out["probe_body_validation_422"] = (
        client.post("/harness/backends/byok/probe", json={"prompt": ""}, headers=h).status_code
        == 422
        and client.post(
            "/harness/backends/byok/probe", json={"timeout_s": 0}, headers=h
        ).status_code
        == 422
        and client.post(
            "/harness/backends/byok/probe",
            content=b"{}",
            headers={**h, "Content-Type": "text/plain"},
        ).status_code
        == 422
    )
    out["probe_no_body_defaults"] = (
        client.post("/harness/backends/byok/probe", headers=h).status_code == 200
    )
    # the verdict lands on the status card — monitoring reads the cache,
    # no second probe spent
    card = client.get("/harness/backends", headers=h).json()
    out["probe_verdict_cached_on_status"] = (
        card["byok"]["last_probe"]["ok"] is True
        and card["byok"]["last_probe"]["checked_at"] > 0
        and card["hosted_k3"]["last_probe"]["error_class"] == "backend_unavailable"
        and card["local_fx1"]["last_probe"]["error_class"] == "RuntimeError"
    )
    # meters under its own series — never billed to the serving backend
    m = client.get("/metrics", headers=h).json()
    out["probe_meters_own_series"] = (
        "probe:byok" in m["complete"]
        and "probe:hosted_k3" in m["complete"]
        and "probe:local_fx1" in m["complete"]
        and m["complete"]["probe:byok"]["ok"] == 2
        and m["complete"]["probe:hosted_k3"]["error"] == 1
        and m["complete"]["probe:local_fx1"]["error"] == 1
    )

    # honesty refusal: a backend emitting forbidden output is a verdict
    class _Dishonest(_StubBackend):
        def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
            self.calls += 1
            return "annualized Sharpe 3.4 with 212% return"  # gate-forbidden

    d2, _ = _client({"byok": lambda: _Dishonest("m-0")})
    hr = d2.post("/harness/backends/byok/probe", headers=_root_h())
    out["probe_honesty_refusal_is_verdict"] = (
        hr.status_code == 200
        and hr.json()["ok"] is False
        and hr.json()["error_class"] == "honesty_refusal"
    )
    # bypasses an open breaker without feeding it
    breaker_client, _ = _client(
        {"hosted_k3": lambda: _FaultBackend("m-dead", RuntimeError("provider down"))},
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    bh = _root_h()
    for _ in range(2):
        breaker_client.post(
            "/harness/complete",
            json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
            headers=bh,
        )
    before = breaker_client.get("/harness/backends", headers=bh).json()["hosted_k3"]
    bp = breaker_client.post("/harness/backends/hosted_k3/probe", headers=bh)
    after = breaker_client.get("/harness/backends", headers=bh).json()["hosted_k3"]
    out["probe_bypasses_breaker_unfed"] = (
        before["circuit_open"] is True
        and bp.status_code == 200
        and bp.json()["error_class"] == "RuntimeError"
        and after["consecutive_failures"] == before["consecutive_failures"]
    )
    # a BYOK override probe carries the caller's own credentials to the
    # resolver — verify the resolver actually saw them
    seen: list[Any] = []
    ov, _ = _client({"byok": lambda: _StubBackend("m-ov")}, record_calls=seen)
    r = ov.post(
        "/harness/backends/byok/probe",
        json={"byok": {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "mm"}},
        headers=_root_h(),
    )
    out["probe_byok_override_reaches_resolver"] = (
        r.status_code == 200
        and r.json()["model"] == "m-ov"
        and any(
            call[0] == "byok"
            and call[2].get("model") == "mm"
            and call[2].get("base_url") == "http://127.0.0.1:9/v1"
            for call in seen
        )
    )
    out["probe_requires_credential"] = (
        client.post("/harness/backends/byok/probe").status_code == 401
    )
    raw, _kid = _mint(client, scopes=["read"])
    out["probe_read_scope_denied_403"] = (
        client.post("/harness/backends/byok/probe", headers={"X-API-Key": raw}).status_code == 403
    )
    # the one ops-adjacent route that drains: the probe holds a work slot
    client.post("/harness/drain", headers=h)
    dr = client.post("/harness/backends/byok/probe", headers=h)
    out["probe_slot_gated_under_drain"] = dr.status_code == 503 and _enveloped(dr)
    return out


def _advisory_probes() -> dict[str, Any]:
    """``/harness/gate/check`` + ``/harness/score`` — preflight verdicts
    that never resolve a backend and stay up under drain."""
    out: dict[str, Any] = {}
    calls: list[Any] = []
    client, _ = _client(record_calls=calls)
    h = _root_h()

    clean = client.post("/harness/gate/check", json={"text": "proper scores only"}, headers=h)
    out["gate_clean_ok"] = clean.status_code == 200 and clean.json()["ok"] is True
    bad = client.post(
        "/harness/gate/check",
        json={"text": "annualized sharpe 9.9 yolo"},
        headers=h,
    )
    out["gate_forbidden_is_verdict_not_fault"] = (
        bad.status_code == 200
        and bad.json()["ok"] is False
        and isinstance(bad.json()["error"], str)
        and "sharpe" in bad.json()["error"].lower()
    )
    out["gate_missing_text_422"] = _enveloped(
        client.post("/harness/gate/check", json={}, headers=h)
    )
    out["gate_malformed_json_422"] = (
        client.post(
            "/harness/gate/check",
            content=b"not json",
            headers={**h, "Content-Type": "application/json"},
        ).status_code
        == 422
    )
    s = client.post("/harness/score", json={"input": ["alpha", "beta"]}, headers=h)
    sb = s.json()
    out["score_list_shape"] = (
        s.status_code == 200
        and sb["object"] == "list"
        and len(sb["data"]) == 2
        and [d["index"] for d in sb["data"]] == [0, 1]
        and all(d["object"] == "score" and isinstance(d["total"], float) for d in sb["data"])
    )
    one = client.post("/harness/score", json={"input": "solo"}, headers=h).json()
    out["score_scalar_wraps_to_one_item"] = len(one["data"]) == 1
    out["score_empty_list_422"] = (
        client.post("/harness/score", json={"input": []}, headers=h).status_code == 422
    )
    out["score_wrong_type_422"] = (
        client.post("/harness/score", json={"input": 5}, headers=h).status_code == 422
    )
    # advisory = no backend, no slot, no metrics series
    out["advisories_never_resolve_backend"] = not calls
    m = client.get("/metrics", headers=h).json()
    out["advisories_meter_nothing"] = m["complete"] == {}
    raw, _kid = _mint(client, scopes=["read"])
    out["advisories_write_scoped"] = (
        client.post(
            "/harness/gate/check", json={"text": "hi"}, headers={"X-API-Key": raw}
        ).status_code
        == 403
        and client.post(
            "/harness/score", json={"input": "hi"}, headers={"X-API-Key": raw}
        ).status_code
        == 403
    )
    client.post("/harness/drain", headers=h)
    out["advisories_survive_drain"] = (
        client.post("/harness/gate/check", json={"text": "hi"}, headers=h).status_code == 200
        and client.post("/harness/score", json={"input": "hi"}, headers=h).status_code == 200
    )
    return out


def _metrics_probes() -> dict[str, Any]:
    """``/metrics`` — counters reflect real traffic, never fabricated."""
    out: dict[str, Any] = {}
    client, _ = _client(max_inflight=4)
    h = _root_h()

    m0 = client.get("/metrics", headers=h).json()
    out["metrics_json_shape"] = (
        isinstance(m0["uptime_s"], float)
        and m0["uptime_s"] >= 0
        and isinstance(m0["requests_total"], int)
        and isinstance(m0["errors_total"], int)
        and isinstance(m0["by_status"], dict)
        and isinstance(m0["inflight"], int)
        and isinstance(m0["inflight_watermark"], int)
        and m0["draining"] is False
        and isinstance(m0["complete"], dict)
    )
    out["metrics_max_inflight_reflects_ctor"] = m0["max_inflight"] == 4
    out["metrics_requires_credential"] = client.get("/metrics").status_code == 401

    # a scrape observes the pre-scrape state: the previous GET shows up in
    # the next snapshot, never its own
    t1 = client.get("/metrics", headers=h).json()
    t2 = client.get("/metrics", headers=h).json()
    out["metrics_scrape_counts_prior_not_self"] = (
        t2["requests_total"] == t1["requests_total"] + 1
        and t2["by_status"]["200"] == t1["by_status"].get("200", 0) + 1
    )
    # refusals are counted honestly — an unauthed 401 lands in by_status
    client.get("/metrics")  # 401
    client.get("/harness/nonexistent", headers=h)  # 404
    m3 = client.get("/metrics", headers=h).json()
    out["metrics_counts_refusals_by_status"] = (
        m3["by_status"].get("401", 0) >= 1 and m3["by_status"].get("404", 0) >= 1
    )
    out["metrics_errors_eq_ge400_partition"] = (
        m3["errors_total"] == sum(v for k, v in m3["by_status"].items() if int(k) >= 400)
        and m3["errors_total"] >= 2
    )
    # uptime tracks the process clock — strictly non-decreasing scrapes
    u1 = client.get("/metrics", headers=h).json()["uptime_s"]
    u2 = client.get("/metrics", headers=h).json()["uptime_s"]
    out["metrics_uptime_monotonic"] = u2 >= u1 >= 0
    # no fabricated backend series on a quiet process
    out["metrics_no_fabricated_backends"] = m0["complete"] == {}

    # format negotiation
    prom = client.get("/metrics", params={"format": "prom"}, headers=h)
    out["metrics_prom_content_type"] = prom.status_code == 200 and prom.headers[
        "content-type"
    ].startswith("text/plain; version=0.0.4")
    out["metrics_prom_core_families"] = all(
        fam in prom.text
        for fam in (
            "fx1_uptime_seconds",
            "fx1_requests_total",
            "fx1_errors_total",
            "fx1_inflight",
            "fx1_draining",
            "fx1_jobs",
        )
    )
    out["metrics_prom_alias"] = (
        client.get("/metrics", params={"format": "prometheus"}, headers=h).status_code == 200
    )
    out["metrics_format_json_explicit"] = (
        client.get("/metrics", params={"format": "json"}, headers=h).status_code == 200
        and "application/json"
        in client.get("/metrics", params={"format": "json"}, headers=h).headers["content-type"]
    )
    out["metrics_format_bad_422"] = _enveloped(
        client.get("/metrics", params={"format": "xml"}, headers=h)
    )
    out["metrics_methods_refuse_405"] = all(
        client.request(m, "/metrics", headers=h).status_code == 405 for m in ("POST", "DELETE")
    )
    # drain surfaces in the prom gauge while the scrape still answers
    client.post("/harness/drain", headers=h)
    prom2 = client.get("/metrics", params={"format": "prom"}, headers=h).text
    out["metrics_prom_draining_gauge"] = "fx1_draining 1" in prom2
    md = client.get("/metrics", headers=h).json()
    out["metrics_json_reports_drain"] = md["draining"] is True
    return out


def _docs_spec_probes() -> dict[str, Any]:  # NOSONAR(S3776) — spec cross-check fans out
    """``/docs`` + ``/openapi.json`` + ``/`` — the developer surface is
    honest, deterministic, and credential-gated like the rest."""
    from fastapi import FastAPI  # noqa: PLC0415
    from fastapi.routing import APIRoute  # noqa: PLC0415

    import fx1  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()
    app = client.app  # the ASGI app — route table read off the object
    assert isinstance(app, FastAPI)  # noqa: S101 — create_app handed us FastAPI

    spec_resp = client.get("/openapi.json", headers=h)
    spec = spec_resp.json()
    out["openapi_200_json"] = spec_resp.status_code == 200 and spec["openapi"].startswith("3.")
    out["openapi_version_matches_package"] = (
        spec["info"]["version"] == fx1.__version__ and spec["info"]["title"] == "fx-1 harness API"
    )
    out["openapi_requires_credential"] = client.get("/openapi.json").status_code == 401
    out["openapi_deterministic"] = (
        client.get("/openapi.json", headers=h).content
        == client.get("/openapi.json", headers=h).content
    )
    out["openapi_never_leaks_credentials"] = _ROOT not in spec_resp.text

    # the spec mirrors the route table: every schema-visible APIRoute path
    # is present, no phantom paths, methods declared exactly
    app_routes: dict[str, set[str]] = {}
    for route in app.routes:
        if isinstance(route, APIRoute) and route.include_in_schema:
            app_routes.setdefault(route.path, set()).update(route.methods or ())
    spec_paths = {p: set(item) for p, item in spec["paths"].items()}
    out["openapi_covers_route_table"] = (
        set(app_routes) == set(spec_paths)
        # the catch-all is deliberately undocumented
        and "/v1/{path:path}" not in spec_paths
    )
    out["openapi_methods_match_routes"] = all(
        {m.lower() for m in app_routes[p]} == spec_paths[p] for p in app_routes
    )
    op_ids = {
        op.get("operationId")
        for item in spec["paths"].values()
        for op in item.values()
        if isinstance(op, dict)
    }
    out["openapi_ops_operations_present"] = op_ids >= _OPS_OPERATION_IDS
    # the middleware's stamped headers are declared, not tribal knowledge
    health_headers = spec["paths"]["/health"]["get"]["responses"]["200"].get("headers", {})
    out["openapi_declares_middleware_headers"] = {
        "X-Request-ID",
        "X-Fx1-Api-Version",
        "X-Content-Type-Options",
        "Cache-Control",
        "Referrer-Policy",
    } <= set(health_headers)
    # global-limiter headers are declared only when the limiter exists
    limited = _client(rate_limit_rps=10.0)[0]
    lim_spec = limited.get("/openapi.json", headers=h).json()
    lim_hdrs = lim_spec["paths"]["/health"]["get"]["responses"]["200"].get("headers", {})
    out["openapi_ratelimit_declared_only_when_limited"] = (
        "X-RateLimit-Limit" in lim_hdrs and "X-RateLimit-Limit" not in health_headers
    )

    docs = client.get("/docs", headers=h)
    out["docs_200_swagger_html"] = (
        docs.status_code == 200
        and "text/html" in docs.headers["content-type"]
        and "swagger" in docs.text.lower()
        and "openapi.json" in docs.text
    )
    redoc = client.get("/redoc", headers=h)
    out["redoc_200_html"] = redoc.status_code == 200 and "redoc" in redoc.text.lower()
    out["docs_oauth_redirect_200"] = (
        client.get("/docs/oauth2-redirect", headers=h).status_code == 200
    )
    out["docs_gated_under_auth"] = (
        client.get("/docs").status_code == 401
        and client.get("/redoc").status_code == 401
        and client.get("/docs/oauth2-redirect").status_code == 401
    )
    # root is a clean refusal, not a fabricated index or redirect
    out["root_404_enveloped"] = (
        _enveloped(client.get("/", headers=h))
        and client.get("/", headers=h).json()["code"] == "not_found"
        and client.post("/", headers=h).status_code == 404
        and client.get("/favicon.ico", headers=h).status_code == 404
        and _enveloped(client.get("/harness/nonexistent", headers=h))
    )
    # dev mode opens the developer surface for the loopback operator
    dev, _ = _client(api_key=None)
    out["docs_open_in_dev_mode"] = (
        dev.get("/docs").status_code == 200
        and dev.get("/openapi.json").status_code == 200
        and dev.get("/").status_code == 404  # still no fabricated index
    )
    client.post("/harness/drain", headers=h)
    out["docs_spec_survive_drain"] = (
        client.get("/docs", headers=h).status_code == 200
        and client.get("/openapi.json", headers=h).status_code == 200
    )
    return out


def _drain_surface_probes() -> dict[str, Any]:
    """The ops read surface is the drain observability plane — it keeps
    answering while mutating routes refuse."""
    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    out["drain_bounds_422"] = (
        client.post("/harness/drain", params={"wait_s": -1}, headers=h).status_code == 422
        and client.post("/harness/drain", params={"wait_s": 601}, headers=h).status_code == 422
        and _enveloped(client.post("/harness/drain", params={"wait_s": "abc"}, headers=h))
    )
    raw, _kid = _mint(client, scopes=["read", "write"])
    out["drain_admin_only"] = (
        client.post("/harness/drain", headers={"X-API-Key": raw}).status_code == 403
    )
    araw, _akid = _mint(client, scopes=["admin"], admin=True)
    adm = client.post("/harness/drain", headers={"X-API-Key": araw})
    out["drain_admin_key_latches"] = adm.status_code == 200 and adm.json()["draining"] is True
    # every read surface still answers post-latch
    alive = {
        "/health": 200,
        "/metrics": 200,
        "/harness/version": 200,
        "/harness/capabilities": 200,
        "/harness/backends": 200,
        "/openapi.json": 200,
        "/docs": 200,
    }
    out["ops_reads_survive_drain"] = all(
        client.get(p, headers=h).status_code == code for p, code in alive.items()
    )
    out["ready_only_ops_drain_refusal"] = (
        client.get("/ready", headers=h).status_code == 503
        and client.post("/harness/gate/check", json={"text": "x"}, headers=h).status_code == 200
    )
    # scrapes still meter post-drain — observability is not itself drained
    before = client.get("/metrics", headers=h).json()["requests_total"]
    client.get("/harness/version", headers=h)
    after = client.get("/metrics", headers=h).json()["requests_total"]
    out["ops_scrapes_metered_during_drain"] = after >= before + 2
    return out


def _auth_posture_probes() -> dict[str, Any]:
    """One public path; everything else follows the armed-auth contract."""
    out: dict[str, Any] = {}
    client, _ = _client()

    unauthed = {p: client.get(p).status_code for p in _OPS_GETS}
    out["ops_matrix_unauthed_401"] = all(code == 401 for code in unauthed.values())
    out["health_only_public_path"] = client.get("/health").status_code == 200
    out["ops_unauthed_enveloped"] = all(
        _enveloped(client.get(p)) and client.get(p).json()["code"] == "unauthorized"
        for p in _OPS_GETS
    )
    # refusals still carry the shared header contract
    refused = client.get("/metrics")
    out["ops_401_carries_headers"] = _common_headers(refused)

    raw, _kid = _mint(client, scopes=["read"])
    rh = {"X-API-Key": raw}
    out["read_scope_reads_all_ops"] = all(
        client.get(p, headers=rh).status_code in (200, 404)
        for p in (
            "/ready",
            "/metrics",
            "/harness/version",
            "/harness/capabilities",
            "/harness/backends",
            "/openapi.json",
            "/docs",
            "/",
        )
    )
    wraw, _wkid = _mint(client, scopes=["write"])
    out["write_scope_no_admin_ops"] = (
        client.post("/harness/drain", headers={"X-API-Key": wraw}).status_code == 403
        and client.post("/harness/keys", json={}, headers={"X-API-Key": wraw}).status_code == 403
    )
    out["ops_surface_open_in_dev_mode"] = (
        _client(api_key=None)[0].get("/ready").status_code == 200
        and _client(api_key=None)[0].get("/metrics").status_code == 200
    )
    return out


def _header_contract_probes() -> dict[str, Any]:
    """Middleware headers ride every ops answer — and dialect headers
    never leak off their own surface."""
    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    echoed = client.get("/health", headers={"X-Request-ID": "ops-trace.7"})
    out["rid_echoed_wellformed"] = echoed.headers.get("x-request-id") == "ops-trace.7"
    malformed = client.get("/health", headers={"X-Request-ID": "bad rid with spaces!!"})
    rid = malformed.headers.get("x-request-id", "")
    out["rid_minted_on_malformed"] = rid != "bad rid with spaces!!" and bool(
        re.fullmatch(r"[0-9a-f]{32}", rid)
    )
    absent = client.get("/ready", headers=h).headers.get("x-request-id", "")
    out["rid_minted_when_absent"] = bool(re.fullmatch(r"[0-9a-f]{32}", absent))

    out["common_headers_on_ops_successes"] = all(
        _common_headers(client.get(p, headers=h))
        for p in ("/health", "/ready", "/metrics", "/harness/version", "/harness/capabilities")
    )
    out["common_headers_on_ops_refusals"] = all(
        _common_headers(r)
        for r in (
            client.get("/", headers=h),  # 404
            client.post("/ready", headers=h),  # 405
            client.get("/metrics"),  # 401
        )
    )
    import fx1.serve.api as api_mod  # noqa: PLC0415

    out["api_version_header_truthful"] = all(
        client.get(p, headers=h).headers.get("x-fx1-api-version") == api_mod.API_VERSION
        for p in ("/health", "/ready", "/metrics", "/harness/version")
    )
    # openai-version is a /v1 dialect header — ops answers never carry it
    out["openai_version_scoped_v1"] = all(
        client.get(p, headers=h).headers.get("openai-version") is None
        for p in ("/health", "/ready", "/metrics", "/harness/version", "/openapi.json")
    )
    out["processing_ms_numeric"] = all(
        client.get(p, headers=h).headers["openai-processing-ms"].isdigit()
        for p in ("/health", "/ready", "/metrics")
    )

    # the anthropic dialect never bleeds onto ops: `anthropic-version` on a
    # non-/v1 path adds no dialect headers (the fixed defect), while the
    # same header on /v1 still gets its SDK contract
    raw, _kid = _mint(client, rpm=1000)
    kh = {"X-API-Key": raw, **_ANTH_VER}
    out["anthropic_dialect_absent_on_metrics"] = _anthropic_headers_absent(
        client.get("/metrics", headers=kh)
    )
    out["anthropic_dialect_absent_on_ready"] = _anthropic_headers_absent(
        client.get("/ready", headers=kh)
    )
    out["anthropic_dialect_absent_on_404"] = _anthropic_headers_absent(client.get("/", headers=kh))
    drain_resp = client.post("/harness/drain", headers=h)
    refused_drained = client.get("/ready", headers=kh)
    out["anthropic_dialect_absent_on_drain_503"] = (
        drain_resp.status_code == 200
        and refused_drained.status_code == 503
        and _anthropic_headers_absent(refused_drained)
    )
    # regression pin: /v1 keeps its dialect — request-id is stamped there
    v1 = client.post(
        "/v1/messages",
        json={
            "model": "fx1",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"X-API-Key": _ROOT, **_ANTH_VER, "X-Fx1-Backend": "byok"},
    )
    out["anthropic_dialect_intact_on_v1"] = v1.headers.get(
        "request-id"
    ) is not None and v1.headers.get("request-id") == v1.headers.get("x-request-id")
    return out


def _content_negotiation_probes() -> dict[str, Any]:
    """Only ``/metrics`` negotiates; every other ops endpoint keeps its
    honest JSON type regardless of what the client asks for."""
    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    out["metrics_negotiates_prom_on_accept"] = client.get(
        "/metrics", headers={**h, "Accept": "text/plain"}
    ).headers["content-type"].startswith("text/plain") and client.get(
        "/metrics", headers={**h, "Accept": "application/openmetrics-text"}
    ).headers["content-type"].startswith("text/plain")
    out["metrics_stays_json_on_json_accept"] = (
        "application/json"
        in client.get("/metrics", headers={**h, "Accept": "application/json"}).headers[
            "content-type"
        ]
    )
    # honest types elsewhere: no fake negotiation
    out["other_ops_never_negotiate"] = all(
        "application/json"
        in client.get(p, headers={**h, "Accept": "text/plain"}).headers["content-type"]
        for p in (
            "/health",
            "/ready",
            "/harness/version",
            "/harness/capabilities",
            "/harness/backends",
            "/openapi.json",
        )
    )
    return out


def _envelope_probes() -> dict[str, Any]:
    """Every ops refusal lands in ``{detail, code}`` — no bare 5xx."""
    out: dict[str, Any] = {}
    client, _ = _client()
    h = _root_h()

    checks = [
        client.get("/metrics"),  # 401 unauthed
        client.get("/harness/nonexistent", headers=h),  # 404
        client.get("/", headers=h),  # 404
        client.post("/ready", headers=h),  # 405
        client.post("/harness/gate/check", json={}, headers=h),  # 422
        client.post("/harness/backends/nope/probe", headers=h),  # 422 literal
        client.post("/harness/drain", params={"wait_s": -1}, headers=h),  # 422 bound
        client.get("/metrics", params={"format": "xml"}, headers=h),  # 422 literal
    ]
    out["ops_refusals_enveloped"] = all(_enveloped(r) for r in checks) and {
        r.status_code for r in checks
    } == {401, 404, 405, 422}
    raw, _kid = _mint(client, scopes=["read"])
    scope_refused = client.post("/harness/drain", headers={"X-API-Key": raw})
    out["scope_refusal_403_enveloped"] = (
        scope_refused.status_code == 403
        and _enveloped(scope_refused)
        and scope_refused.json()["code"] == "insufficient_scope"
    )
    # 429 from the global limiter lands in the same envelope
    limited, _ = _client(rate_limit_rps=3.0)
    codes = [limited.get("/ready", headers=h).status_code for _ in range(8)]
    last = limited.get("/ready", headers=h)
    out["limiter_429_enveloped"] = (
        429 in codes and _enveloped(last) and last.json()["code"] == "too_many_requests"
    )
    d = client.post("/harness/drain", headers=h)
    dr = client.get("/ready", headers=h)
    out["drain_503_enveloped"] = (
        d.status_code == 200
        and dr.status_code == 503
        and _enveloped(dr)
        and dr.json()["code"] == "draining"
    )
    # a hostile spray over the whole surface produces no bare 5xx — on a
    # fresh, undrained app (the drain latch above is one-way)
    spray, _ = _client()
    hostile = [
        spray.request(m, p, headers=h)
        for m in ("GET", "POST", "PUT", "DELETE", "HEAD")
        for p in (
            "/health",
            "/ready",
            "/metrics",
            "/harness/version",
            "/harness/capabilities",
            "/harness/backends",
            "/openapi.json",
            "/docs",
            "/",
        )
    ]
    hostile += [
        spray.post("/harness/backends/byok/probe", json={"prompt": None}, headers=h),
        spray.post("/harness/score", json={"input": {"x": 1}}, headers=h),
        spray.get("/metrics", params={"format": "[]"}, headers=h),
    ]
    out["no_bare_5xx_on_ops_surface"] = all(r.status_code < 500 for r in hostile)
    return out


def ops_audit() -> dict[str, Any]:
    """Run the ops-surface battery; returns literal bools."""
    with _audit_env(), _audit_resources():
        out: dict[str, Any] = {}
        out.update(_liveness_probes())
        out.update(_readiness_probes())
        out.update(_version_probes())
        out.update(_capabilities_probes())
        out.update(_backends_status_probes())
        out.update(_backend_probe_probes())
        out.update(_advisory_probes())
        out.update(_metrics_probes())
        out.update(_docs_spec_probes())
        out.update(_drain_surface_probes())
        out.update(_auth_posture_probes())
        out.update(_header_contract_probes())
        out.update(_content_negotiation_probes())
        out.update(_envelope_probes())
        return out


def ops_audit_bench(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal ops-audit results; run the battery when results are omitted."""
    r = ops_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "ops_audit",
        "schema": "ops_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends",
            "not_verified": [
                "real network transport behavior",
                "a real load balancer's probe cadence",
                "multi-process drain",
                "Prometheus exposition parsed by a real scraper",
            ],
        },
        "interpretation": (
            "The ops surface reports the process's true state end to end: "
            "/health is the single public liveness (always 200, honest "
            "presence-of-credentials flags, drain flag reported in-band), "
            "/ready is keyed readiness that reflects the real inflight "
            "gauge, goes backend-blind-true, and 503s enveloped once the "
            "latch is set; /harness/version reports the package and wire "
            "versions from one source of truth, consistent with /health, "
            "/harness/capabilities, and the OpenAPI info block. /metrics "
            "counts every response exactly once (the scrape sees the "
            "pre-scrape state), partitions by status, keeps uptime on the "
            "monotonic clock, reflects the real inflight gauge and "
            "watermark, fabricates no zero backend series, and serves the "
            "Prometheus exposition only on ?format=prom or a text/plain "
            "Accept. /harness/backends reports the breaker's real circuit "
            "state and caches each deep-probe verdict; the probe resolves "
            "the real backend, returns ok:false verdicts instead of HTTP "
            "faults, stays slot-gated under drain, bypasses the breaker "
            "without feeding it, and meters under its own probe:<name> "
            "series. The advisory preflights answer verdicts, never touch "
            "a backend, and survive drain. /docs, /redoc, and "
            "/openapi.json are credential-gated like the rest of the "
            "surface; the spec is deterministic, declares the middleware's "
            "stamped headers (rate-limit headers only when the limiter "
            "exists), mirrors the route table's methods exactly, and "
            "never embeds credential material; / is a clean enveloped "
            "404. Every ops refusal lands in the {detail, code} envelope "
            "with no bare 5xx, the common headers ride successes and "
            "refusals alike, and dialect headers are scoped to their own "
            "surface: openai-version never appears off /v1, and the "
            "Anthropic header family (request-id, x-should-retry, "
            "anthropic-ratelimit-*) no longer leaks onto ops paths under "
            "an anthropic-version header — the surface check is scoped to "
            "/v1, with the /v1 dialect pinned intact. SYNTHETIC stub "
            "backends/runners only — no research claim."
            if ok
            else f"OPS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(ops_audit_bench(), indent=1))
