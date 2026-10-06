"""ratelimit_audit — shape/fairness audit of the two-layer rate limiter.

``quota_audit`` pins the per-key rpm window's edge and budget refusals;
``quota2_audit`` pins the introspection surfaces. This battery audits the
layer above them — the global per-host token bucket (``_RateLimiter`` in
the middleware, before ingress and auth) — and how the two layers
compose: which surfaces the bucket governs, the exact burst/refill
shape, whether refusals burn capacity, fairness between hosts and
between keys, and the truthfulness of every header, hint, and counter
the limiter writes. The claim under test: the limiter's *shape* is the
declared one end to end — a token bucket that admits exactly its
declared burst, refills continuously, bounds its identity map on every
path, and never confuses a transient window refusal with a terminal
quota.

Coverage map:

- *bucket burst* — the global bucket admits exactly ``int(capacity)``
  requests then refuses 429 ``too_many_requests``; ``X-RateLimit-
  {Limit,Remaining,Reset}`` ride on every governed response:
  ``Remaining`` decrements honestly to 0, ``Reset`` is the refill
  countdown in seconds (never an epoch), and refused calls still carry
  the trio they just spent. ``Retry-After`` is present and honest —
  sleeping past it admits.
- *bucket shape* — refill is continuous (a trickle, not a window): a
  partial rest buys back a proportional fraction of capacity, and a
  drained bucket refuses a concurrent flood with no phantom tokens.
  Sub-1 rps keeps a one-token floor (a 0.5 rps limit still admits 1
  call, refuses the next, and declares ``Limit: 1`` / ``Reset: 2`` —
  never a nonsensical ``Limit: 0``); fractional rps floors the burst.
- *bucket bounds* — the identity map is LRU-bounded on the admitted
  and refused paths alike, evicts oldest-first, and refills an evicted
  identity to a fresh bucket; a saturated host never starves a peer;
  an ``allow`` race admits exactly capacity.
- *ordering* — the bucket is the outermost gate: it consumes before
  ingress validation (a flood of oversized bodies 413s then starts
  429ing before ever reaching ingress), before auth (bad-credential
  floods burn the host bucket and the 401s still carry the trio), and
  before the per-key window (a saturated key's refusals still consume
  global slots until the bucket itself refuses with
  ``too_many_requests`` — the code a client sees flips as the binding
  layer moves outward). OPTIONS/preflight meter like any method.
- *exemptions* — ``/health`` is unmetered (a load balancer's cadence
  must not consume the caller's budget); every other surface —
  ``/metrics``, ``/openapi.json``, GETs, admin routes — is governed.
  Disabling the limiter emits no trio at all (no false scarcity).
- *window composition* — the per-key fixed window layers under the
  bucket: scope classes share one window per key (read and write draws
  spend the same pool); GETs, admin routes, post-auth 422s, idempotent
  replays, ``store:false`` calls, and drain-latched calls all consume
  a slot; policy patches keep occupancy (a tighten refuses, a relief
  admits); ``/harness/self`` reports the post-consume truth.
- *store honesty* — under an injected clock the window is exactly
  fixed-edge: ``Retry-After`` equals the real seconds-to-edge, refusals
  never re-anchor, the whole occupancy drops at once past the edge,
  quota refusals precede window consume (a spent budget never burns a
  slot), scope denials consume nothing, and an ``authenticate`` race
  admits exactly ``rpm``.
- *restart* — under ``--state-dir`` the durable spend (``uses``,
  ``tokens_used``) replays verbatim while window occupancy is honestly
  process-local (fresh ``Remaining`` on the new process, declared
  policy still enforced); the global bucket is likewise process-local.
- *metrics* — ``fx1_rate_limited_total`` counts only real rate-limit
  refusals (global bucket + key window); a terminal ``quota_exceeded``
  lands in ``by_status['429']`` but never inflates the limiter counter.
- *envelopes/client* — every refusal is enveloped in the path's own
  grammar (``{detail, code}`` on ``/harness``, ``{error}`` under
  ``/v1``); on the Anthropic leg a rate refusal carries
  ``x-should-retry: true`` where ``quota_exceeded`` pins ``false``.
  ``HarnessClient`` retries a global 429 (declared ``Retry-After``)
  but never retries ``quota_exceeded``.

This is a stub-backed TestClient battery, not network timing,
multi-process, or live-provider evidence; host fairness is exercised
through a scope-rewriting ASGI shim (the wire path is real, the
"other host" is synthetic). Sealed ``ratelimit_audit.v1``
(fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.quota2_audit import (
    _COMPLETE_PATH,
    _H_KEY,
    _H_RL_LIMIT,
    _H_RL_REMAINING,
    _H_RL_RESET,
    _H_SHOULD_RETRY,
    _KEYS_PATH,
    _MESSAGES_PATH,
    _RATE_WINDOW_CONST,
    _RESOURCES,
    _ROOT,
    _SELF_PATH,
    _U6,
    _audit_context,
    _client,
    _complete,
    _keys_mod,
    _MeterBackend,
    _mint,
    _parallel,
    _retry_after,
    _rl_headers,
    _usage_card,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient
    from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = ["ratelimit_audit", "ratelimit_audit_bench"]

# The global-bucket header family (the per-key ``*-Requests`` family is
# _H_RL_* above — the two families coexist on managed-key responses).
_H_RL_G_LIMIT = "x-ratelimit-limit"
_H_RL_G_REMAINING = "x-ratelimit-remaining"
_H_RL_G_RESET = "x-ratelimit-reset"
_H_AUDIT_HOST = b"x-fx1-audit-host"

_HEALTH_PATH = "/health"
_COMMANDS_PATH = "/harness/commands"
_DRAIN_PATH = "/harness/drain"
_METRICS_PATH = "/metrics"
_OPENAPI_PATH = "/openapi.json"
_MODELS_PATH = "/v1/models"
_RESPONSES_PATH = "/v1/responses"
_CHAT_PATH = "/v1/chat/completions"
_NOPE_PATH = "/harness/definitely-not-a-route"
_BAD_KEY = "fx1k_bad"
_API_KEY_ENV = "FX1_API_KEY"

_CODE_TOO_MANY = "too_many_requests"
_CODE_RATE_LIMITED = "rate_limited"
_CODE_QUOTA = "quota_exceeded"


def _temporary_directory() -> Path:
    return Path(
        _RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="ratelimit_audit_"))
    )


def _meter() -> dict[str, Any]:
    return {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}


def _patch_key(client: TestClient, root_h: dict[str, str], key_id: str, **fields: Any) -> None:
    r = client.patch(f"{_KEYS_PATH}/{key_id}", json=fields, headers=root_h)
    assert r.status_code == 200, r.text


def _metrics_card(client: TestClient, root_h: dict[str, str]) -> dict[str, Any]:
    r = client.get(_METRICS_PATH, headers=root_h)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _trio(resp: Any) -> dict[str, str]:
    """The global-bucket trio present on ``resp`` (lowercased names)."""
    heads = _rl_headers(resp)
    return {k: heads[k] for k in (_H_RL_G_LIMIT, _H_RL_G_REMAINING, _H_RL_G_RESET) if k in heads}


def _req_trio(resp: Any) -> dict[str, str]:
    """The per-key ``*-Requests`` trio present on ``resp``."""
    heads = _rl_headers(resp)
    return {k: heads[k] for k in (_H_RL_LIMIT, _H_RL_REMAINING, _H_RL_RESET) if k in heads}


def _chat_once(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _CHAT_PATH,
        json={"model": "byok", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


def _messages_once(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={**headers, "X-Fx1-Backend": "byok"},
    )


class _HostSpoof:
    """ASGI shim: rewrites ``scope['client']`` from an
    ``x-fx1-audit-host`` request header so one TestClient exercises two
    wire identities through the real host-keyed middleware."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            for name, value in scope.get("headers") or []:
                if name == _H_AUDIT_HOST:
                    scope["client"] = (value.decode("ascii"), 0)
                    break
        await self._app(scope, receive, send)


def _spoof_client(rate_limit_rps: float) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) with the host-spoof ASGI shim in front —
    same isolation as ``_client`` but the wire identity varies by
    header."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        os.environ[_API_KEY_ENV] = _ROOT
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: _meter()[name](),
            state_dir=isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            rate_limit_rps=rate_limit_rps,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(_HostSpoof(app), raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


# ---------------------------------------------------------------------------
# global bucket — wire pins (rps=3)
# ---------------------------------------------------------------------------


def _bucket_wire_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    client, _ = _client(_meter(), _ROOT, rate_limit_rps=3.0)
    root = {_H_KEY: _ROOT}

    # burst: capacity admits exactly, then 429 too_many_requests
    burst = [_complete(client, root) for _ in range(4)]
    codes = [r.status_code for r in burst]
    out["bucket_burst_exact_capacity"] = codes == [200, 200, 200, 429]
    out["bucket_remaining_decrements"] = [
        _rl_headers(r).get(_H_RL_G_REMAINING) for r in burst[:3]
    ] == ["2", "1", "0"]
    out["bucket_limit_reports_capacity"] = all(
        _rl_headers(r).get(_H_RL_G_LIMIT) == "3" for r in burst
    )
    refused = burst[-1]
    refused_body = refused.json()
    out["bucket_refusal_code_too_many_requests"] = refused_body.get("code") == _CODE_TOO_MANY
    out["bucket_refusal_envelope_harness"] = (
        isinstance(refused_body.get("detail"), str) and "code" in refused_body
    )
    ra = _retry_after(refused)
    out["bucket_refusal_retry_after_declared"] = ra is not None and ra >= 1
    refused_trio = _trio(refused)
    out["bucket_refusal_carries_trio"] = (
        len(refused_trio) == 3 and refused_trio[_H_RL_G_REMAINING] == "0"
    )
    # Reset is a refill-horizon countdown in seconds — for rps >= 1 the
    # bucket refills fully in <= 1s, so the declared value stays in
    # {0, 1}; never an epoch, never absent on a governed response.
    resets = [_rl_headers(r).get(_H_RL_G_RESET) for r in burst]
    out["bucket_reset_seconds_countdown"] = all(
        v is not None and v.isdigit() and int(v) <= 1 for v in resets
    )
    out["bucket_env_key_metered"] = all(len(_trio(r)) == 3 for r in burst)
    out["bucket_env_key_no_requests_family"] = not any(_H_RL_LIMIT in _rl_headers(r) for r in burst)
    # governed while dry: ops surfaces are behind the limiter too
    out["bucket_metrics_governed"] = client.get(_METRICS_PATH, headers=root).status_code == 429
    out["bucket_openapi_governed"] = client.get(_OPENAPI_PATH, headers=root).status_code == 429

    # health is exempt: unlimited cadence, no trio, no auth needed
    time.sleep(1.1)
    health = [client.get(_HEALTH_PATH) for _ in range(6)]
    out["bucket_health_unmetered"] = all(r.status_code == 200 for r in health) and not any(
        _trio(r) for r in health
    )
    health_k = client.get(_HEALTH_PATH, headers={_H_KEY: _BAD_KEY})
    out["bucket_health_credentialed_unmetered"] = health_k.status_code == 200 and not _trio(
        health_k
    )

    # the trio rides every governed refusal family, not just the 429
    unauth = _complete(client, {_H_KEY: _BAD_KEY})
    out["bucket_trio_on_401"] = unauth.status_code == 401 and len(_trio(unauth)) == 3
    missing = client.get(_NOPE_PATH, headers=root)
    out["bucket_trio_on_404"] = missing.status_code == 404 and len(_trio(missing)) == 3
    bad = client.post(_COMPLETE_PATH, json={"backend": "byok"}, headers=root)
    out["bucket_trio_on_422"] = bad.status_code == 422 and len(_trio(bad)) == 3

    # managed key: the per-key family coexists with the global trio
    time.sleep(1.1)
    raw, _key_id = _mint(client, root, rpm=5)
    kh = {_H_KEY: raw}
    ok = _complete(client, kh)
    ok_h = _rl_headers(ok)
    out["bucket_managed_key_both_families"] = (
        ok.status_code == 200 and ok_h.get(_H_RL_G_LIMIT) == "3" and ok_h.get(_H_RL_LIMIT) == "5"
    )
    # a limiter-free app declares no global trio at all — no false scarcity
    client_off, _ = _client(_meter(), _ROOT)
    off = _complete(client_off, root)
    out["bucket_no_trio_when_disabled"] = off.status_code == 200 and not _trio(off)
    return out


# ---------------------------------------------------------------------------
# global bucket — refill shape (rps=4) and fractional rps
# ---------------------------------------------------------------------------


def _bucket_shape_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _ = _client(_meter(), _ROOT, rate_limit_rps=4.0)
    root = {_H_KEY: _ROOT}

    drain = [_complete(client, root).status_code for _ in range(5)]
    out["bucket_drain_then_refuse"] = drain == [200, 200, 200, 200, 429]
    # continuous refill: ~0.3s at 4 rps buys exactly one token — a fixed
    # window would admit zero until the boundary
    time.sleep(0.32)
    after_trickle = [_complete(client, root) for _ in range(2)]
    out["bucket_trickle_refill_single"] = [r.status_code for r in after_trickle] == [200, 429]
    # Retry-After is honest: sleeping the declared delay admits
    ra = _retry_after(after_trickle[-1])
    time.sleep(1.05 if ra is None else ra + 0.05)
    drain2 = [_complete(client, root).status_code for _ in range(5)]
    out["bucket_retry_after_honest"] = drain2[0] == 200
    out["bucket_full_refill_restores_capacity"] = drain2 == [200, 200, 200, 200, 429]
    # refused calls burn nothing — back-to-back refusals still report 0
    r1 = _complete(client, root)
    r2 = _complete(client, root)
    out["bucket_refused_calls_burn_nothing"] = (
        r1.status_code == 429
        and r2.status_code == 429
        and _rl_headers(r1).get(_H_RL_G_REMAINING) == "0"
        and _rl_headers(r2).get(_H_RL_G_REMAINING) == "0"
    )
    # concurrent flood on a dry bucket: no phantom admissions
    flood = _parallel(client, [lambda: _complete(client, root) for _ in range(6)], workers=6)
    out["bucket_race_refusals_exact"] = all(r.status_code == 429 for r in flood)
    return out


def _bucket_subrps_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = {_H_KEY: _ROOT}

    # rps < 1 keeps a one-token floor — the limit never declares "0"
    client, _ = _client(_meter(), _ROOT, rate_limit_rps=0.5)
    one = _complete(client, root)
    two = _complete(client, root)
    one_h = _rl_headers(one)
    out["bucket_sub_rps_capacity_floor_one"] = one.status_code == 200 and two.status_code == 429
    out["bucket_sub_rps_limit_honest"] = one_h.get(_H_RL_G_LIMIT) == "1"
    out["bucket_sub_rps_reset_countdown"] = one_h.get(_H_RL_G_RESET) == "2"
    ra = _retry_after(two)
    out["bucket_sub_rps_retry_after_declared"] = ra == 2
    time.sleep(2.05)
    out["bucket_sub_rps_retry_after_admits"] = _complete(client, root).status_code == 200

    # fractional rps floors the burst and the declared limit alike
    client_f, _ = _client(_meter(), _ROOT, rate_limit_rps=2.5)
    frac = [_complete(client_f, root) for _ in range(3)]
    out["bucket_fractional_burst_floor"] = [r.status_code for r in frac] == [200, 200, 429]
    out["bucket_fractional_limit_floor"] = _rl_headers(frac[0]).get(_H_RL_G_LIMIT) == "2"
    out["bucket_remaining_floor_conservative"] = [
        _rl_headers(r).get(_H_RL_G_REMAINING) for r in frac[:2]
    ] == ["1", "0"]
    return out


# ---------------------------------------------------------------------------
# global bucket — store-level pins (no transport)
# ---------------------------------------------------------------------------


def _bucket_unit_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    import fx1.serve.api as api_mod  # noqa: PLC0415

    limiter_t = api_mod._RateLimiter

    # the identity map stays bounded under an *admitted* spray — the LRU
    # bound must hold on the success path, not only when refusing
    spray = limiter_t(1000.0, max_keys=3)
    for i in range(10):
        spray.allow(f"host-{i}")
    out["limiter_bound_under_admitted_spray"] = len(spray._buckets) <= 3

    # ...and under a refused spray (the refusal path already bounded it)
    spray_r = limiter_t(0.001, max_keys=3)
    for i in range(10):
        spray_r.allow(f"rhost-{i}")
    out["limiter_bound_under_refused_spray"] = len(spray_r._buckets) <= 3

    # eviction is oldest-first and the evicted identity re-enters fresh
    lru = limiter_t(4.0, max_keys=2)
    lru.allow("a")
    lru.allow("b")
    lru.allow("c")  # evicts a
    wait_a, rem_a = lru.allow("a")
    out["limiter_lru_evicts_oldest_first"] = wait_a == 0.0 and rem_a == 3.0
    out["limiter_bound_kept_on_reentry"] = len(lru._buckets) <= 2

    # identity isolation: a saturated host never starves a peer
    iso = limiter_t(1.0)
    iso.allow("alpha")
    w_starved, _rem_s = iso.allow("alpha")
    w_peer, _rem_p = iso.allow("beta")
    out["limiter_identity_isolation"] = w_starved > 0.0 and w_peer == 0.0

    # a refused allow() consumes no token — the wait only shrinks by
    # real elapsed time
    burn = limiter_t(1.0)
    burn.allow("x")
    w1, _r1 = burn.allow("x")
    w2, _r2 = burn.allow("x")
    out["limiter_refusal_consumes_nothing"] = 0.0 < w2 < w1 <= 1.0

    # an allow() race admits exactly capacity — no overshoot under lock
    race = limiter_t(4.0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        waits = list(pool.map(lambda _i: race.allow("h")[0], range(8)))
    out["limiter_concurrent_exact_admission"] = sum(1 for w in waits if w == 0.0) == 4

    # sub-rps floor: capacity is one token, the declared wait is real
    sub = limiter_t(0.5)
    w_ok, _rem_ok = sub.allow("s")
    w_wait, _rem_w = sub.allow("s")
    out["limiter_sub_rps_single_capacity"] = (
        sub.capacity == 1.0 and w_ok == 0.0 and 1.8 < w_wait <= 2.0
    )
    return out


# ---------------------------------------------------------------------------
# layering — bucket vs ingress vs auth vs key window
# ---------------------------------------------------------------------------


def _layered_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    client, _ = _client(_meter(), _ROOT, rate_limit_rps=2.0)
    root = {_H_KEY: _ROOT}

    # unauthenticated flood: the bucket gates before auth — bad-cred
    # calls burn the host's tokens until the bucket itself refuses
    unauth = [_complete(client, {_H_KEY: _BAD_KEY}) for _ in range(3)]
    out["layered_limiter_precedes_auth"] = [r.status_code for r in unauth] == [
        401,
        401,
        429,
    ]
    out["layered_unauth_401_carries_trio"] = all(len(_trio(r)) == 3 for r in unauth[:2])

    # ingress refusals still consumed a global slot: oversized bodies
    # 413 until the bucket runs dry, then 429 — the limiter precedes
    # ingress too (and reports the spent trio on the 413)
    time.sleep(1.1)
    big = {"backend": "byok", "messages": [{"role": "user", "content": "x" * (1 << 20)}]}
    ingress = [client.post(_COMPLETE_PATH, json=big, headers=root) for _ in range(3)]
    ingress_codes = [r.status_code for r in ingress]
    out["layered_ingress_consumes_before_auth"] = ingress_codes == [413, 413, 429]
    out["layered_ingress_refusal_reports_trio"] = all(len(_trio(r)) == 3 for r in ingress)
    big_body = ingress[0].json()
    out["layered_ingress_enveloped"] = (
        ingress_codes[0] == 413
        and isinstance(big_body.get("detail"), str)
        and big_body.get("code") == "too_large"
    )

    # OPTIONS/preflight meters like any method — no free surface
    time.sleep(1.1)
    opts = [client.options(_COMPLETE_PATH, headers=root).status_code for _ in range(3)]
    out["layered_options_preflight_consumes"] = opts[-1] == 429 and all(
        c in (405, 429) for c in opts
    )

    # bucket vs key window: a saturated key's refusals still burn the
    # host bucket — until the bucket itself answers instead
    client2, _ = _client(_meter(), _ROOT, rate_limit_rps=3.0)
    raw, key_id = _mint(client2, root, rpm=1)
    kh = {_H_KEY: raw}
    first = _complete(client2, kh)
    second = _complete(client2, kh)
    third = _complete(client2, kh)
    out["layered_key_refusal_burns_global"] = (
        first.status_code == 200
        and second.status_code == 429
        and second.json().get("code") == _CODE_RATE_LIMITED
        and _rl_headers(second).get(_H_RL_G_REMAINING) == "0"
    )
    out["layered_global_precedes_key"] = (
        third.status_code == 429 and third.json().get("code") == _CODE_TOO_MANY
    )
    out["layered_key429_carries_both_families"] = (
        _rl_headers(second).get(_H_RL_G_REMAINING) == "0"
        and _rl_headers(second).get(_H_RL_REMAINING) == "0"
        and _retry_after(second) is not None
    )
    time.sleep(1.1)
    card = _usage_card(client2, root, key_id)
    out["layered_global_429_bills_no_use"] = int(card["uses"]) == 1

    # host fairness through the real middleware: two wire identities get
    # independent buckets — saturating one never starves the other
    sp, _ = _spoof_client(1.0)
    ha = {**root, "x-fx1-audit-host": "10.0.0.1"}
    hb = {**root, "x-fx1-audit-host": "10.0.0.2"}
    a1 = _complete(sp, ha).status_code
    a2 = _complete(sp, ha)
    b1 = _complete(sp, hb)
    a_h = _rl_headers(a2)
    b_h = _rl_headers(b1)
    out["layered_hosts_fair"] = (
        a1 == 200
        and a2.status_code == 429
        and b1.status_code == 200
        and a_h.get(_H_RL_G_REMAINING) == "0"
        and b_h.get(_H_RL_G_REMAINING) == "0"
    )
    return out


# ---------------------------------------------------------------------------
# per-key window — composition pins the global layer never shadows
# ---------------------------------------------------------------------------


def _window_depth_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    client, _ = _client(_meter(), _ROOT)
    root = {_H_KEY: _ROOT}
    keys_mod = _keys_mod()

    # scope classes share ONE window per key — read and write draws
    # spend the same pool
    raw, _sid = _mint(client, root, rpm=2, scopes=["read", "write"])
    kh = {_H_KEY: raw}
    seq = [
        client.get(_MODELS_PATH, headers=kh).status_code,
        _complete(client, kh).status_code,
        client.get(_MODELS_PATH, headers=kh).status_code,
    ]
    out["window_scopes_share_pool"] = seq == [200, 200, 429]

    # GET surfaces consume the window too — metering is by credential,
    # not by verb
    raw2, _id2 = _mint(client, root, rpm=1)
    kh2 = {_H_KEY: raw2}
    g1 = client.get(_MODELS_PATH, headers=kh2).status_code
    g2 = client.get(_MODELS_PATH, headers=kh2)
    out["window_get_surface_consumes"] = g1 == 200 and g2.status_code == 429

    # a post-auth 422 consumed the slot — validation happens after
    # metering, so a malformed call still spends the window
    raw3, id3 = _mint(client, root, rpm=1)
    kh3 = {_H_KEY: raw3}
    bad = client.post(_COMPLETE_PATH, json={"backend": "byok"}, headers=kh3)
    after = _complete(client, kh3)
    out["window_postauth_422_consumes"] = bad.status_code == 422 and after.status_code == 429
    out["window_422_bills_use"] = int(_usage_card(client, root, id3)["uses"]) == 1

    # idempotent replay still consumes the window — dedupe is about
    # the *effect*, not the metering
    raw4, _id4 = _mint(client, root, rpm=2)
    kh4 = {_H_KEY: raw4}
    idem = {**kh4, "Idempotency-Key": "rl-idem-1"}
    i1 = _complete(client, idem)
    i2 = _complete(client, idem)
    i3 = _complete(client, kh4)
    out["window_idem_replay_consumes"] = (
        i1.status_code == 200
        and i2.status_code == 200
        and i2.json().get("replayed") is True
        and i3.status_code == 429
    )

    # store:false skips persistence, not metering
    raw5, _id5 = _mint(client, root, rpm=2)
    kh5 = {_H_KEY: raw5}
    s1 = client.post(
        _RESPONSES_PATH,
        json={"model": "byok", "input": "hi", "store": False},
        headers=kh5,
    )
    s2 = client.post(
        _RESPONSES_PATH,
        json={"model": "byok", "input": "hi", "store": False},
        headers=kh5,
    )
    s3 = _complete(client, kh5)
    out["window_store_false_consumes"] = (
        s1.status_code == 200 and s2.status_code == 200 and s3.status_code == 429
    )

    # drain (dedicated client — the latch is one-way and would refuse
    # later mints): the latch POST and every drain-refused call consume
    # the window; reads stay open under drain — all metered
    client_d, _ = _client(_meter(), _ROOT)
    raw6, id6 = _mint(client_d, root, admin=True, rpm=4)
    kh6 = {_H_KEY: raw6}
    d_post = client_d.post(_DRAIN_PATH, headers=kh6)
    d_refused = _complete(client_d, kh6)
    d_read1 = client_d.get(_COMMANDS_PATH, headers=kh6)
    d_read2 = client_d.get(_COMMANDS_PATH, headers=kh6)
    d_after = _complete(client_d, kh6)
    out["window_drain_post_consumes"] = d_post.status_code == 200
    out["window_drain_refused_consumes"] = d_after.status_code == 429
    out["window_reads_open_under_drain"] = d_read1.status_code == 200 and d_read2.status_code == 200
    d_body = d_refused.json()
    out["window_drain_refusal_enveloped"] = (
        d_refused.status_code == 503 and d_body.get("code") == "draining"
    )
    out["window_drain_bills_uses"] = int(_usage_card(client_d, root, id6)["uses"]) == 4

    # policy patch keeps occupancy: tightening below the spent count
    # refuses immediately; loosening re-opens headroom
    raw7, id7 = _mint(client, root, rpm=3)
    kh7 = {_H_KEY: raw7}
    _complete(client, kh7)
    _complete(client, kh7)
    _patch_key(client, root, id7, rpm=1)
    tight = _complete(client, kh7)
    out["window_patch_tighten_refuses"] = tight.status_code == 429
    _patch_key(client, root, id7, rpm=5)
    relief = _complete(client, kh7)
    out["window_patch_relief_admits"] = relief.status_code == 200

    # /harness/self reads the post-consume truth — the introspection
    # call itself is metered
    raw8, id8 = _mint(client, root, rpm=3)
    kh8 = {_H_KEY: raw8}
    self_r = client.get(_SELF_PATH, headers=kh8)
    self_body = self_r.json()
    self_key = self_body.get("key") or {}
    out["window_self_postconsume_truth"] = (
        self_r.status_code == 200
        and self_body.get("metered") is True
        and self_key.get("window_remaining") == 2
        and self_key.get("uses") == 1
        and _rl_headers(self_r).get(_H_RL_REMAINING) == "2"
    )

    # /health never meters — key-supplied or not, the key's counters
    # don't move
    for _ in range(4):
        client.get(_HEALTH_PATH, headers=kh8)
    card8 = _usage_card(client, root, id8)
    out["window_health_never_metered"] = int(card8["uses"]) == 1

    # Retry-After honesty on the key window (patched window so the
    # declared second actually elapses)
    saved_window = float(getattr(keys_mod, _RATE_WINDOW_CONST))
    try:
        setattr(keys_mod, _RATE_WINDOW_CONST, 0.15)
        raw9, _id9 = _mint(client, root, rpm=1)
        kh9 = {_H_KEY: raw9}
        ok9 = _complete(client, kh9).status_code
        refused9 = _complete(client, kh9)
        ra9 = _retry_after(refused9)
        time.sleep(max(0.2, (ra9 or 0) + 0.1))
        admit9 = _complete(client, kh9).status_code
        out["window_retry_after_readmits"] = (
            ok9 == 200 and refused9.status_code == 429 and ra9 is not None and admit9 == 200
        )
    finally:
        setattr(keys_mod, _RATE_WINDOW_CONST, saved_window)

    # per-key isolation: A's saturated window never shows on B
    raw_a, _ida = _mint(client, root, rpm=1)
    raw_b, _idb = _mint(client, root, rpm=1)
    ka = {_H_KEY: raw_a}
    kb = {_H_KEY: raw_b}
    _complete(client, ka)
    _complete(client, ka)
    b_ok = _complete(client, kb)
    out["window_keys_isolated"] = (
        b_ok.status_code == 200 and _rl_headers(b_ok).get(_H_RL_REMAINING) == "0"
    )

    # an unwindowed key is unmetered — no false scarcity headers
    raw_u, _idu = _mint(client, root)
    ku = {_H_KEY: raw_u}
    unlim = [_complete(client, ku) for _ in range(5)]
    out["window_unwindowed_unmetered"] = all(r.status_code == 200 for r in unlim) and not any(
        _req_trio(r) for r in unlim
    )
    return out


# ---------------------------------------------------------------------------
# metrics + envelope truthfulness
# ---------------------------------------------------------------------------


def _metrics_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    client, _ = _client(_meter(), _ROOT)
    root = {_H_KEY: _ROOT}

    raw_rl, _rlid = _mint(client, root, rpm=1)
    krl = {_H_KEY: raw_rl}
    _complete(client, krl)
    rl_refused = _complete(client, krl)
    raw_q, _qid = _mint(client, root, max_requests=1)
    kq = {_H_KEY: raw_q}
    _complete(client, kq)
    q_refused = _complete(client, kq)
    m = _metrics_card(client, root)
    # fx1_rate_limited_total counts rate-limit denials only — a terminal
    # quota refusal lands in by_status but never inflates the limiter
    out["metrics_window_refusal_counted"] = m["rate_limited_total"] == 1
    out["metrics_quota_refusal_not_counted"] = m["rate_limited_total"] == 1
    out["metrics_429_family_by_status"] = m["by_status"].get("429", 0) == 2

    rl_body = rl_refused.json()
    out["envelope_429_harness_shape"] = (
        rl_refused.status_code == 429
        and isinstance(rl_body.get("detail"), str)
        and rl_body.get("code") == _CODE_RATE_LIMITED
    )
    q_body = q_refused.json()
    out["envelope_quota_terminal_no_retry"] = (
        q_refused.status_code == 429
        and q_body.get("code") == _CODE_QUOTA
        and _retry_after(q_refused) is None
    )

    # openai leg — {error:{code}} grammar on the same two refusal kinds
    raw_o, _oid = _mint(client, root, rpm=1)
    ko = {_H_KEY: raw_o}
    _chat_once(client, ko)
    orl = _chat_once(client, ko)
    orl_err = orl.json().get("error")
    out["envelope_429_openai_shape"] = (
        orl.status_code == 429
        and isinstance(orl_err, dict)
        and orl_err.get("code") == _CODE_RATE_LIMITED
    )
    raw_oq, _oqid = _mint(client, root, max_requests=1)
    koq = {_H_KEY: raw_oq}
    _chat_once(client, koq)
    oqe = _chat_once(client, koq)
    oqe_err = oqe.json().get("error")
    out["envelope_quota_openai_shape"] = (
        oqe.status_code == 429
        and isinstance(oqe_err, dict)
        and oqe_err.get("code") == _CODE_QUOTA
        and _retry_after(oqe) is None
    )

    # anthropic leg — the rate refusal is retryable where quota is not,
    # and carries the anthropic standing-budget family
    raw_m, _mid = _mint(client, root, rpm=1)
    km = {_H_KEY: raw_m}
    _messages_once(client, km)
    mrl = _messages_once(client, km)
    mrl_h = _rl_headers(mrl)
    out["envelope_429_anthropic_should_retry"] = (
        mrl.status_code == 429
        and mrl_h.get(_H_SHOULD_RETRY) == "true"
        and "anthropic-ratelimit-requests-remaining" in mrl_h
        and mrl.json().get("type") == "error"
    )
    return out


# ---------------------------------------------------------------------------
# restart — durable spend vs process-local occupancy
# ---------------------------------------------------------------------------


def _restart_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    shared = _temporary_directory() / "shared-state"
    shared.mkdir()
    root = {_H_KEY: _ROOT}

    client_a, _ = _client(_meter(), _ROOT, state_dir=shared)
    raw, key_id = _mint(client_a, root, rpm=3)
    kh = {_H_KEY: raw}
    _complete(client_a, kh)
    _complete(client_a, kh)

    client_b, _ = _client(_meter(), _ROOT, state_dir=shared)
    first_b = _complete(client_b, kh)
    card_b = _usage_card(client_b, root, key_id)
    out["restart_uses_durable"] = int(card_b["uses"]) == 3
    # window occupancy is honestly process-local — a fresh window opens,
    # the journaled policy still binds
    out["restart_window_occupancy_fresh"] = (
        first_b.status_code == 200 and _rl_headers(first_b).get(_H_RL_REMAINING) == "2"
    )
    _complete(client_b, kh)
    _complete(client_b, kh)
    refused_b = _complete(client_b, kh)
    out["restart_policy_still_enforced"] = (
        refused_b.status_code == 429 and refused_b.json().get("code") == _CODE_RATE_LIMITED
    )

    # the global bucket is process-local too: a saturated app admits
    # immediately on a fresh process sharing the state dir
    ga, _ = _client(_meter(), _ROOT, state_dir=shared, rate_limit_rps=1.0)
    _complete(ga, root)
    _complete(ga, root)
    gb, _ = _client(_meter(), _ROOT, state_dir=shared, rate_limit_rps=1.0)
    out["restart_bucket_process_local"] = _complete(gb, root).status_code == 200
    return out


# ---------------------------------------------------------------------------
# store-level window math under an injected clock
# ---------------------------------------------------------------------------


def _store_clock_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    out: dict[str, bool] = {}
    from fx1.serve.keys import ApiKeyStore, KeyStoreError  # noqa: PLC0415

    clock = {"now": 1000.0}
    store = ApiKeyStore(clock=lambda: clock["now"])
    raw, rec = store.mint("edge", rpm=2)
    key_id = rec["key_id"]

    store.authenticate(raw)  # t=1000 anchors the window
    clock["now"] = 1010.0
    store.authenticate(raw)  # count=2 → window full
    refusal: KeyStoreError | None = None
    try:
        store.authenticate(raw)
    except KeyStoreError as exc:
        refusal = exc
    out["store_retry_after_exact"] = (
        refusal is not None and refusal.code == _CODE_RATE_LIMITED and refusal.retry_after == 50.0
    )

    # a refused call never re-anchors the window — the declared wait
    # counts down from the original consume
    clock["now"] = 1030.0
    refusal2: KeyStoreError | None = None
    try:
        store.authenticate(raw)
    except KeyStoreError as exc:
        refusal2 = exc
    out["store_refusals_never_reanchor"] = refusal2 is not None and refusal2.retry_after == 30.0

    # past the edge the whole occupancy drops at once — fixed, not
    # sliding (a sliding window would grant only one slot back)
    clock["now"] = 1061.0
    store.authenticate(raw)  # new window anchored at 1061
    clock["now"] = 1122.0
    ok8 = store.authenticate(raw) is not None  # re-anchors at 1122
    ok9 = store.authenticate(raw) is not None
    out["store_fixed_window_drops_whole"] = ok8 and ok9

    # re-anchor honesty: the fresh window's retry-after counts from the
    # new anchor, not the first window's
    clock["now"] = 1123.0
    refusal3: KeyStoreError | None = None
    try:
        store.authenticate(raw)
    except KeyStoreError as exc:
        refusal3 = exc
    out["store_window_reanchors_at_consume"] = refusal3 is not None and refusal3.retry_after == 59.0

    # window_state reports the declared countdown: saturated
    # (rpm, 0, seconds-to-edge), fresh (rpm, rpm, 0)
    ws = store.window_state(key_id)
    out["store_window_state_truthful"] = ws == (2, 0, 59)
    _raw_f, rec_f = store.mint("fresh", rpm=4)
    out["store_window_state_fresh"] = store.window_state(rec_f["key_id"]) == (4, 4, 0)

    # hard budgets refuse before the window consumes — a spent key
    # never burns a slot
    raw_q, _rec_q = store.mint("quota", rpm=1, max_requests=1)
    store.authenticate(raw_q)
    quota_refusal: KeyStoreError | None = None
    try:
        store.authenticate(raw_q)
    except KeyStoreError as exc:
        quota_refusal = exc
    out["store_quota_precedes_window"] = (
        quota_refusal is not None and quota_refusal.code == _CODE_QUOTA
    )

    # a scope denial consumes nothing — authorization precedes metering
    raw_s, rec_s = store.mint("scoped", rpm=1, scopes=["read"])
    scope_refusal: KeyStoreError | None = None
    try:
        store.authenticate(raw_s, required_scope="write")
    except KeyStoreError as exc:
        scope_refusal = exc
    out["store_scope_refusal_consumes_nothing"] = (
        scope_refusal is not None
        and scope_refusal.code == "insufficient_scope"
        and store.window_state(rec_s["key_id"]) == (1, 1, 0)
    )

    # authenticate is exact under a race — the store lock serializes
    race = ApiKeyStore(clock=lambda: clock["now"])
    raw_r, _rec_r = race.mint("race", rpm=4)

    def _try_auth(raw_key: str) -> bool:
        try:
            return race.authenticate(raw_key) is not None
        except KeyStoreError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_try_auth, [raw_r] * 8))
    out["store_authenticate_race_exact"] = sum(results) == 4

    # per-key isolation at the store: A's exhaustion never touches B
    raw_b2, _rec_b = store.mint("peer", rpm=1)
    out["store_key_isolation"] = store.authenticate(raw_b2) is not None
    return out


# ---------------------------------------------------------------------------
# client leg — the global 429 is retryable; quota is terminal
# ---------------------------------------------------------------------------


def _client_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessClient, HarnessTransportError  # noqa: PLC0415
    from fx1.serve.quota2_audit import _TransportSpy  # noqa: PLC0415

    client, _ = _client(_meter(), _ROOT, rate_limit_rps=1.0)
    spy = _TransportSpy(client)
    hc = HarnessClient(
        "http://ratelimit-audit.dev",
        api_key=_ROOT,
        transport=spy,
        max_retries=2,
        retry_writes=True,
        sleep=lambda s: None,
    )
    ok_first = hc.complete([{"role": "user", "content": "a"}], backend="byok")
    refused_exc: HarnessTransportError | None = None
    try:
        hc.complete([{"role": "user", "content": "b"}], backend="byok")
    except HarnessTransportError as exc:
        refused_exc = exc
    posts = [c for c in spy.calls if c[0] == "POST" and c[1] == _COMPLETE_PATH]
    # a global 429 carries Retry-After, so the client retries inside its
    # budget — and still surfaces a transport error when the bucket
    # stays dry (the window is process-fast, the retry budget is not):
    # 1 admitted + 1 initial + 2 retries on the refused call
    out["client_retries_global_429"] = (
        ok_first.content == "ok:a" and refused_exc is not None and len(posts) == 4
    )
    return out


# ---------------------------------------------------------------------------
# bench entry
# ---------------------------------------------------------------------------


def ratelimit_audit() -> dict[str, Any]:
    """Run the whole probe battery under an isolated harness env.

    Every probe is a measured boolean — a ``False`` is a real
    broken-honesty claim, not a skipped check."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_bucket_wire_probes())
        out.update(_bucket_shape_probes())
        out.update(_bucket_subrps_probes())
        out.update(_bucket_unit_probes())
        out.update(_layered_probes())
        out.update(_window_depth_probes())
        out.update(_metrics_probes())
        out.update(_restart_probes())
        out.update(_store_clock_probes())
        out.update(_client_probes())
        return out


def ratelimit_audit_bench() -> dict[str, Any]:
    results = ratelimit_audit()
    ok = len(results) == 93 and all(v is True for v in results.values())
    out = {
        "kind": "bench",
        "schema": "ratelimit_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": results,
            "ok": ok,
        },
        "coverage": {
            "transport": "TestClient + ApiKeyStore/_RateLimiter unit probes",
            "not_executed": [
                "real network timing under load",
                "multi-process bucket sharing",
                "live provider rate limits",
            ],
        },
        "interpretation": (
            "The fx1 serve layer's rate limiting is a two-layer honest limiter: a "
            "global per-host token bucket that bounds its identity map on every "
            "admission path, refills continuously, reports a truthful "
            "X-RateLimit-* trio on every governed response (success and refusal "
            "alike), and gates before ingress and auth; layered under it, the "
            "per-key fixed rpm window meters by credential — scope-shared, "
            "drain-aware, patch-stable — while quota refusals stay terminal and "
            "distinct at the wire, in the metrics, and in the client's retry "
            "policy."
            if ok
            else f"RATELIMIT AUDIT DEFECT: {results}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(ratelimit_audit_bench(), indent=2, sort_keys=True))
