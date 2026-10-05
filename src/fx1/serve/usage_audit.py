"""usage_audit — adversarial probes on usage-accounting conservation.

The claim under test: the numbers ``GET /harness/usage`` reports are
*conserved* — every completion record is counted exactly once, every
provider-reported token is billed to exactly the credential that spent
it, filters partition the retained window disjointly, and a truncated
ring declares its evictions instead of silently losing them.

Coverage map:

- *Conservation* — scripted completions with known stub usage, then
  every published field re-derived independently: per-backend,
  per-model, and per-key buckets each sum back to ``totals``, and
  ``totals`` equals a fold over the raw completion records.
- *Filter partition* — ``backend``/``model``/``key_id`` slices are
  disjoint and reconstruct the whole; ``since``/``until`` bound the
  record's own ``at`` inclusively on both edges, probed at the exact
  boundary timestamp. ``since > until`` fails closed 400.
- *Counter reconciliation* — a managed key's ``uses`` counts
  *authenticated requests* (not completions: a request that fails after
  auth still counts; a request refused by a budget, a rate window, a
  scope check, or bad credentials never does). ``tokens_used`` equals
  the per-record charge formula — ``total_tokens`` else
  ``prompt+completion`` — summed over the key's retained records, plus
  the batch-level ``usage_total`` delta that is deliberately charged
  off-ring. Streaming and non-streaming bill identically; idempotent
  replays never double-bill.
- *Ring honesty* — overflow evicts oldest-first, ``records_dropped``
  counts evictions exactly, ``totals`` cover only the retained window,
  and dropped records are still *billed* (the charge fires at append;
  the ring bounds evidence, not spend — a key's ``served`` card is
  declared a lower bound via ``log_dropped``).
- *BYOK vs managed split* — ``backend="byok"`` lands in the ``byok``
  bucket under both the request's credential and the provider's model
  name; a fallback chain attributes the call to the link that actually
  served, never the primary that failed.
- *Concurrency* — batch workers and parallel HTTP posts land every
  record under race; the ring's drop accounting stays exact.
- *Adversarial usage payloads* — a backend that reports non-int, bool,
  negative, or non-dict usage, or no usage at all, can never crash the
  report or corrupt its sums: non-int values are unbillable and dropped
  from every sum, negatives flow verbatim into the buckets (the
  provider's claim, reported faithfully) while ``charge_tokens`` clamps
  them at zero — a budget can never go negative. ``(none)`` bucket
  labels are display-only.

Two defects were found and fixed while building this battery:
``_bucket`` summed canonical usage keys without a type check, so one
poisoned record (``{"prompt_tokens": "many"}``) crashed the whole
billing view with a 500; and a scope-refused request bumped ``uses``
before the authorization check ran, contradicting the documented
"refusals never count" contract. Both are pinned below.

Sealed ``usage_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["usage_audit", "usage_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
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
)
# Deliberately-insecure literal: BYOK overrides accept http:// for local
# stacks — this URL is never dialed (the injected resolver returns stubs).
_BYOK_BASE_URL = "http://127.0.0.1:9/v1"  # NOSONAR(S5332) — never dialed

# Scripted provider usage payloads — the audit's ground truth.
_BYOK_U = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "cached_tokens": 1}
_LOCAL_U = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
_STREAM_U = {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15}


class _MeterBackend:
    """Usage-reporting stub — mirrors ``_UsageTracker``: ``last_usage``
    per call, cumulative ``total_usage`` for the batch delta, plus a
    stream surface that may report a different scripted payload."""

    def __init__(
        self,
        model: str,
        usage: dict[str, int] | None,
        stream_usage: dict[str, int] | None = None,
    ) -> None:
        self._model = model
        self._usage = usage
        self._stream_usage = usage if stream_usage is None else stream_usage
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self.total_usage: dict[str, int] = {}
        self._lock = threading.Lock()

    def _report(self, usage: dict[str, int] | None) -> None:
        self.last_usage = usage
        if not usage:
            return
        with self._lock:
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        self.calls += 1
        self._report(dict(self._usage) if self._usage is not None else None)
        return f"ok:{messages[-1]['content']}"

    def stream(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> Any:
        self.calls += 1
        use = self._stream_usage
        self._report(dict(use) if use is not None else None)
        yield "tok-a"
        yield "tok-b"

    def close(self) -> None:
        pass


class _FailBackend:
    """Never reached the provider: raises before any usage exists."""

    def __init__(self, model: str) -> None:
        self._model = model

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

        raise BackendNotConfiguredError("no credentials configured")

    def close(self) -> None:
        pass


class _AnyUsageBackend:
    """Adversarial stub: hands the wire whatever ``last_usage`` payload it
    is constructed with — non-dict, non-int values, bools, negatives."""

    def __init__(self, model: str, usage: Any) -> None:
        self._model = model
        self._usage = usage
        self.last_usage: Any = None

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        self.last_usage = self._usage
        return "ok"

    def close(self) -> None:
        pass


class _BareBackend:
    """No ``_model`` and no ``last_usage`` attribute at all — the wire
    reads both through getattr defaults."""

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return "ok"

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _FakeRec:
    """Structural ``_Record`` for unit-level aggregation probes — the
    protocol is read-only attributes, so a frozen dataclass conforms."""

    backend: str
    model: str | None
    ok: bool
    latency_ms: float
    at: float
    usage: dict[str, int] | None = None
    key_id: str | None = None


@dataclass
class _Ledger:
    """Expected completion records, appended by the audit as it issues
    calls — the independent fold the wire is reconciled against."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        backend: str,
        model: str | None,
        ok: bool,
        usage: dict[str, int] | None,
        key_id: str,
    ) -> None:
        self.records.append(
            {"backend": backend, "model": model, "ok": ok, "usage": usage, "key_id": key_id}
        )

    def bucket(self, key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self.records:
            tag = r[key] if r[key] is not None else "(none)"
            b = out.setdefault(tag, {"requests": 0, "ok": 0, "tokens": 0, "reported": 0})
            b["requests"] += 1
            b["ok"] += 1 if r["ok"] else 0
            if r["usage"]:
                b["reported"] += 1
                b["tokens"] += int(r["usage"].get("total_tokens") or 0)
        return out


def _charge(usage: dict[str, int] | None) -> int:
    """The wire's per-record billing formula (``_charge_key_tokens``):
    ``total_tokens`` when present, else ``prompt+completion``; non-dict /
    empty usage bills nothing. Re-derived here so the meter is checked
    against an independent statement of the contract."""
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    return (pt if isinstance(pt, int) else 0) + (ct if isinstance(ct, int) else 0)


def _client(
    backend_map: dict[str, Any],
    api_key: str | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` zero-arg factories."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: backend_map[name](),
        )
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _complete(
    client: TestClient,
    backend: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> Any:
    return client.post(
        "/harness/complete",
        json={"backend": backend, "messages": [{"role": "user", "content": "hi"}], **extra},
        headers=headers or {},
    )


def _usage(client: TestClient, headers: dict[str, str], **params: Any) -> dict[str, Any]:
    r = client.get("/harness/usage", params=params, headers=headers)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _key_card(client: TestClient, root_h: dict[str, str], key_id: str) -> dict[str, Any]:
    r = client.get(f"/harness/keys/{key_id}/usage", headers=root_h)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _completions(client: TestClient, root_h: dict[str, str]) -> list[dict[str, Any]]:
    r = client.get("/harness/completions", params={"limit": 256}, headers=root_h)
    assert r.status_code == 200, r.text
    items: list[dict[str, Any]] = r.json()["items"]
    return items


def _scenario() -> dict[str, Any]:
    """The main battery — one app, scripted traffic, every published
    aggregate reconciled against an independent ledger."""
    out: dict[str, Any] = {}
    client, _ = _client(
        {
            "byok": lambda: _MeterBackend("fx1-ft:pilot-9", dict(_BYOK_U), dict(_STREAM_U)),
            "local_fx1": lambda: _MeterBackend("fx1-ckpt-1", dict(_LOCAL_U)),
            "hosted_k3": lambda: _FailBackend("dead-0"),
        },
        api_key=_ROOT,
    )
    root_h = {"X-API-Key": _ROOT}
    k1_raw, k1_id = _mint(client, root_h)
    k1_h = {"X-API-Key": k1_raw}
    led = _Ledger()

    def hit(resp: Any, backend: str, model: str | None, ok: bool, usage: Any, kid: str) -> None:
        if ok:
            assert resp.status_code == 200, resp.text
        led.add(backend, model, ok, usage, kid)

    # --- env-credential traffic ------------------------------------------------
    for _ in range(2):
        hit(_complete(client, "byok", root_h), "byok", "fx1-ft:pilot-9", True, _BYOK_U, "env")
    hit(
        _complete(client, "local_fx1", root_h, checkpoint_dir="synthetic-ckpt"),
        "local_fx1",
        "fx1-ckpt-1",
        True,
        _LOCAL_U,
        "env",
    )
    # --- managed key: sync, SSE stream, anthropic leg, one dead link -----------
    hit(_complete(client, "byok", k1_h), "byok", "fx1-ft:pilot-9", True, _BYOK_U, k1_id)
    rs = client.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
        headers=k1_h,
    )
    hit(rs, "byok", "fx1-ft:pilot-9", True, _STREAM_U, k1_id)
    fail = _complete(client, "hosted_k3", k1_h)
    out["unreached_error_unbilled"] = (
        fail.status_code == 503 and fail.json().get("code") == "backend_unavailable"
    )
    led.add("hosted_k3", None, False, None, k1_id)  # no link served → model None
    ra = client.post(
        "/v1/messages",
        json={
            "model": "fx1",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={**k1_h, "X-Fx1-Backend": "byok"},
    )
    hit(ra, "byok", "fx1-ft:pilot-9", True, _BYOK_U, k1_id)
    # --- n-fanout on the OpenAI leg ---------------------------------------------
    rn = client.post(
        "/v1/chat/completions",
        json={"model": "byok", "n": 2, "messages": [{"role": "user", "content": "hi"}]},
        headers=root_h,
    )
    assert rn.status_code == 200, rn.text
    for _ in range(2):
        led.add("byok", "fx1-ft:pilot-9", True, _BYOK_U, "env")
    # --- batch under its own key ------------------------------------------------
    k6_raw, k6_id = _mint(client, root_h)
    k6_h = {"X-API-Key": k6_raw}
    rb = client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        },
        headers=k6_h,
    )
    assert rb.status_code == 200, rb.text
    batch_usage_total = rb.json().get("usage_total")
    for _ in range(3):
        led.add("byok", "fx1-ft:pilot-9", True, None, k6_id)

    # --- conservation -----------------------------------------------------------
    rep = _usage(client, root_h)
    n = len(led.records)
    out["totals_equal_calls_made"] = rep["records_seen"] == n and rep["totals"]["requests"] == n
    exp_b = led.bucket("backend")
    exp_m = led.bucket("model")
    exp_k = led.bucket("key_id")
    by_b = rep["by_backend"]
    by_m = rep["by_model"]
    by_k = rep["by_key"]

    def _recon(got: dict[str, Any], exp: dict[str, dict[str, Any]]) -> bool:
        return set(got) == set(exp) and all(
            got[t]["requests"] == e["requests"]
            and got[t]["ok"] == e["ok"]
            and got[t]["usage_reported"] == e["reported"]
            and got[t]["total_tokens"] == e["tokens"]
            for t, e in exp.items()
        )

    out["by_backend_sums_match_totals"] = _recon(by_b, exp_b) and (
        sum(b["requests"] for b in by_b.values()) == rep["totals"]["requests"]
    )
    out["by_model_sums_match_totals"] = _recon(by_m, exp_m) and (
        sum(b["requests"] for b in by_m.values()) == rep["totals"]["requests"]
    )
    out["by_key_sums_match_totals"] = _recon(by_k, exp_k) and (
        sum(b["requests"] for b in by_k.values()) == rep["totals"]["requests"]
    )
    # re-derive the totals straight off the raw records — a third fold
    items = _completions(client, root_h)
    exp_prompt = sum(int(r["usage"].get("prompt_tokens") or 0) for r in led.records if r["usage"])
    exp_completion = sum(
        int(r["usage"].get("completion_tokens") or 0) for r in led.records if r["usage"]
    )
    exp_total = sum(int(r["usage"].get("total_tokens") or 0) for r in led.records if r["usage"])
    exp_cached = sum(int(r["usage"].get("cached_tokens") or 0) for r in led.records if r["usage"])
    got_prompt = sum(
        int(r["usage"].get("prompt_tokens") or 0) for r in items if isinstance(r.get("usage"), dict)
    )
    out["totals_rederive_from_records"] = (
        len(items) == n
        and rep["totals"]["prompt_tokens"] == got_prompt == exp_prompt
        and rep["totals"]["completion_tokens"] == exp_completion
        and rep["totals"]["total_tokens"] == exp_total
        and rep["totals"]["other_usage"] == {"cached_tokens": exp_cached}
    )
    out["ok_plus_errors_equals_requests"] = (
        rep["totals"]["ok"] + rep["totals"]["errors"] == rep["totals"]["requests"]
        and rep["totals"]["errors"] == 1
    )
    out["usage_reported_counts_only_reporting"] = (
        rep["totals"]["usage_reported"] == sum(1 for r in led.records if r["usage"]) == n - 4
    )  # the dead-link record + 3 batch items carry no usage dict
    out["mean_latency_is_true_mean"] = rep["totals"]["mean_latency_ms"] == sum(
        r["latency_ms"] for r in items
    ) / len(items)
    # --- filter partition ---------------------------------------------------------
    out["backend_filters_partition"] = all(
        _usage(client, root_h, backend=b)["records_seen"] == exp_b[b]["requests"] for b in exp_b
    )
    out["model_filters_partition"] = all(
        _usage(client, root_h, model=m)["records_seen"] == exp_m[m]["requests"]
        for m in exp_m
        if m != "(none)"
    )
    out["key_filters_partition"] = all(
        _usage(client, root_h, key_id=k)["records_seen"] == exp_k[k]["requests"]
        for k in exp_k
        if k != "(none)"
    )
    ats = sorted(float(r["at"]) for r in items)
    lo, hi = ats[0], ats[-1]
    out["window_counts_match_record_bounds"] = (
        _usage(client, root_h, since=lo, until=hi)["records_seen"] == n
        and _usage(client, root_h, since=hi)["records_seen"] == sum(1 for a in ats if a >= hi)
        and _usage(client, root_h, until=lo)["records_seen"] == sum(1 for a in ats if a <= lo)
        and _usage(client, root_h, since=hi + 1.0)["records_seen"] == 0
        and _usage(client, root_h, until=lo - 1.0)["records_seen"] == 0
    )
    bad = client.get("/harness/usage", params={"since": hi, "until": lo}, headers=root_h)
    neg = client.get("/harness/usage", params={"since": -1.0}, headers=root_h)
    out["window_inverted_400"] = bad.status_code == 400 and bad.json().get("code") == "bad_window"
    out["window_negative_422"] = neg.status_code == 422
    # --- managed-key meters ---------------------------------------------------------
    card1 = _key_card(client, root_h, k1_id)
    k1_charge = sum(_charge(r["usage"]) for r in led.records if r["key_id"] == k1_id)
    out["uses_counts_authenticated_calls"] = (
        card1["uses"] == sum(1 for r in led.records if r["key_id"] == k1_id) == 4
    )
    out["tokens_used_eq_record_charges"] = (
        card1["tokens_used"] == k1_charge == 10 + 15 + 10
        and card1["served"]["total_tokens"] == k1_charge
    )
    out["served_card_matches_ring"] = (
        card1["served"]["calls"] == exp_k[k1_id]["requests"] == 4
        and card1["served"]["by_backend"].get("byok", {}).get("calls") == 3
        and card1["served"]["by_backend"].get("hosted_k3", {}).get("calls") == 1
        and card1["served"]["by_backend"]["hosted_k3"]["total_tokens"] == 0
    )
    out["stream_bills_like_sync"] = by_k[k1_id]["total_tokens"] == k1_charge
    card6 = _key_card(client, root_h, k6_id)
    out["batch_items_carry_no_usage"] = (
        by_k[k6_id]["requests"] == 3 and by_k[k6_id]["usage_reported"] == 0
    )
    out["batch_charges_via_total_delta"] = (
        card6["tokens_used"] == 3 * _BYOK_U["total_tokens"]
        and isinstance(batch_usage_total, dict)
        and batch_usage_total.get("total_tokens") == 3 * _BYOK_U["total_tokens"]
        and card6["served"]["total_tokens"] == 0  # off-ring charge, disclosed
    )
    # --- replay never double-bills ---------------------------------------------------
    before = card1["tokens_used"]
    uses_before = card1["uses"]
    body = {"backend": "byok", "messages": [{"role": "user", "content": "replay-me"}]}
    h = {**k1_h, "Idempotency-Key": "rep-1"}
    r1 = client.post("/harness/complete", json=body, headers=h)
    r2 = client.post("/harness/complete", json=body, headers=h)
    mid = _key_card(client, root_h, k1_id)
    n_items = len(_completions(client, root_h))
    out["replay_never_double_bills"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and mid["tokens_used"] == before + _BYOK_U["total_tokens"]
        and mid["uses"] == uses_before + 2  # both requests authenticated
        and n_items == n + 1  # one new record, not two
    )
    return out


def _refusal_probes() -> dict[str, Any]:
    """Quota/window/scope refusals: denied requests leave no record, burn
    no budget, and bill no tokens."""
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    qraw, qid = _mint(client, root_h, max_requests=1)
    qh = {"X-API-Key": qraw}
    ok_r = _complete(client, "byok", qh)
    denied = _complete(client, "byok", qh)
    card = _key_card(client, root_h, qid)
    rep = _usage(client, root_h, key_id=qid)
    out["quota_denied_no_record_no_use"] = (
        ok_r.status_code == 200
        and denied.status_code == 429
        and denied.json().get("code") == "quota_exceeded"
        and card["uses"] == 1
        and card["tokens_used"] == _BYOK_U["total_tokens"]
        and rep["records_seen"] == 1
    )
    rraw, rid = _mint(client, root_h, rpm=1)
    rh = {"X-API-Key": rraw}
    _complete(client, "byok", rh)
    limited = _complete(client, "byok", rh)
    rcard = _key_card(client, root_h, rid)
    out["rpm_denied_no_record_no_use"] = (
        limited.status_code == 429
        and limited.json().get("code") == "rate_limited"
        and rcard["uses"] == 1
        and _usage(client, root_h, key_id=rid)["records_seen"] == 1
    )
    # the token budget gates the NEXT call — the crossing call completes
    # and is billed, then the key refuses
    traw, tid = _mint(client, root_h, max_tokens=15)
    th = {"X-API-Key": traw}
    c1 = _complete(client, "byok", th)  # 10 reported
    c2 = _complete(client, "byok", th)  # 20 cumulative — crosses 15, still completes
    c3 = _complete(client, "byok", th)  # refused
    tcard = _key_card(client, root_h, tid)
    out["token_budget_gates_next_call"] = (
        c1.status_code == 200
        and c2.status_code == 200
        and c3.status_code == 429
        and c3.json().get("code") == "quota_exceeded"
        and tcard["uses"] == 2
        and tcard["tokens_used"] == 2 * _BYOK_U["total_tokens"]
        and tcard["tokens_remaining"] == 0
        and _usage(client, root_h, key_id=tid)["records_seen"] == 2
    )
    # a read-only key's write attempt is denied 403 and — per the store's
    # contract — never counts as a use, lands no record, touches no meter
    sraw, sid = _mint(client, root_h, scopes=["read"])
    sh = {"X-API-Key": sraw}
    denied_write = _complete(client, "byok", sh)
    safe = client.get("/harness/commands", headers=sh)
    scard = _key_card(client, root_h, sid)
    out["scope_denied_never_bills"] = (
        denied_write.status_code == 403
        and denied_write.json().get("code") == "insufficient_scope"
        and safe.status_code == 200
        and scard["uses"] == 1  # only the admitted read counted
        and scard["tokens_used"] == 0
        and _usage(client, root_h, key_id=sid)["records_seen"] == 0
    )
    # a 401 (bad credential) is not a call at all
    bad = _complete(client, "byok", {"X-API-Key": "fx1k_deadbeef"})
    out["bad_credential_no_record"] = (
        bad.status_code == 401 and _usage(client, root_h)["records_seen"] == 4
    )  # q1 + r1 + t1 + t2
    return out


def _split_probes() -> dict[str, Any]:
    """BYOK vs managed attribution and the fallback serving link."""
    out: dict[str, Any] = {}
    # fallback: primary byok dead → local_fx1 serves; the record must
    # attribute the serving link, not the requested primary
    fb_client, _ = _client(
        {
            "byok": lambda: _FailBackend("byok-dead"),
            "local_fx1": lambda: _MeterBackend("fx1-ckpt-1", dict(_LOCAL_U)),
        },
        api_key=_ROOT,
    )
    root_h = {"X-API-Key": _ROOT}
    r = fb_client.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "fallbacks": ["local_fx1"],
            "checkpoint_dir": "synthetic-ckpt",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=root_h,
    )
    fb_usage = _usage(fb_client, root_h)
    out["fallback_bills_serving_link"] = (
        r.status_code == 200
        and r.json()["backend"] == "local_fx1"
        and fb_usage["by_backend"].get("local_fx1", {}).get("requests") == 1
        and fb_usage["by_backend"]["local_fx1"]["total_tokens"] == _LOCAL_U["total_tokens"]
        and "byok" not in fb_usage["by_backend"]
        and fb_usage["totals"]["errors"] == 0
    )
    # BYOK with per-call override creds lands in the byok bucket and under
    # the calling managed key — provider identity, not credential
    client, _ = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))}, api_key=_ROOT)
    mraw, mid = _mint(client, root_h)
    mh = {"X-API-Key": mraw}
    _complete(client, "byok", root_h)
    _complete(
        client,
        "byok",
        mh,
        byok={"base_url": _BYOK_BASE_URL, "api_key": "dummy", "model": "m"},
    )
    rep = _usage(client, root_h)
    out["byok_calls_land_byok_bucket"] = rep["by_backend"].get("byok", {}).get("requests") == 2
    out["managed_byok_dual_attribution"] = (
        rep["by_key"].get(mid, {}).get("requests") == 1
        and rep["by_key"].get("env", {}).get("requests") == 1
        and rep["by_backend"]["byok"]["total_tokens"] == 2 * _BYOK_U["total_tokens"]
    )
    return out


def _ring_probes() -> dict[str, Any]:
    """Overflow the bounded ring: evictions counted exactly, totals cover
    only the retained window, evicted calls stay billed."""
    out: dict[str, Any] = {}
    client, api_mod = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    raw, kid = _mint(client, root_h)
    kh = {"X-API-Key": raw}
    cap = 256
    n_calls = cap + 4
    for _ in range(n_calls):
        r = _complete(client, "byok", kh)
        assert r.status_code == 200, r.text
    rep = _usage(client, root_h)
    out["ring_cap_declared"] = rep["ring_cap"] == cap
    out["records_dropped_exact"] = rep["records_dropped"] == 4
    out["totals_cover_retained_only"] = (
        rep["records_seen"] == cap
        and rep["totals"]["requests"] == cap
        and rep["totals"]["total_tokens"] == cap * _BYOK_U["total_tokens"]
    )
    # the truncation marker rides every filtered view — ring-level, not
    # filter-level (a narrowed report still discloses the evictions)
    out["dropped_ring_level_on_filtered"] = (
        _usage(client, root_h, backend="byok")["records_dropped"] == 4
        and _usage(client, root_h, key_id=kid)["records_dropped"] == 4
    )
    # dropped records still billed: the charge fired at append; the card
    # declares truncation via log_dropped and reports the lower bound
    card = _key_card(client, root_h, kid)
    out["dropped_records_still_billed"] = (
        card["tokens_used"] == n_calls * _BYOK_U["total_tokens"]
        and card["served"]["calls"] == cap
        and card["served"]["total_tokens"] == cap * _BYOK_U["total_tokens"]
        and card["log_dropped"] == 4
        and card["log_cap"] == cap
    )
    # unit-level: the ring itself counts evictions exactly under a small cap
    log = api_mod._CompletionLog(cap=2)  # noqa: SLF001 — the audit reaches the ring
    for i in range(4):
        log.append(
            api_mod.CompletionRecord(
                completion_id=f"c{i}",
                backend="byok",
                ok=True,
                latency_ms=1.0,
                at=float(i),
                prompt_sha256="p",
            )
        )
    out["ring_unit_drop_exact"] = log.dropped == 2 and len(log.all()) == 2
    return out


def _concurrency_probes() -> dict[str, Any]:
    """Parallel records: batch workers, concurrent HTTP posts, and
    threaded ring appends all keep exact accounting."""
    out: dict[str, Any] = {}
    client, api_mod = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    raw, kid = _mint(client, root_h)
    kh = {"X-API-Key": raw}
    rb = client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": f"q{i}"}] for i in range(16)],
            "max_workers": 8,
        },
        headers=kh,
    )
    items = _completions(client, root_h)
    out["batch_items_all_logged"] = (
        rb.status_code == 200
        and len(rb.json()["results"]) == 16
        and len(items) == 16
        and all(r.get("key_id") == kid for r in items)
    )
    # concurrent HTTP posts — one record each, no loss under contention
    n_par = 24
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _i: _complete(client, "byok", kh).status_code, range(n_par)))
    rep2 = _usage(client, root_h)
    card = _key_card(client, root_h, kid)
    out["parallel_posts_no_lost_records"] = (
        all(c == 200 for c in codes)
        and rep2["records_seen"] == 16 + n_par
        and rep2["by_key"][kid]["requests"] == 16 + n_par
        and card["uses"] == 1 + n_par  # the batch counted once, each post once
        and card["served"]["calls"] == 16 + n_par
    )
    # threaded ring appends: drop accounting exact under contention
    log = api_mod._CompletionLog(cap=64)  # noqa: SLF001 — the audit reaches the ring
    barrier = threading.Barrier(4)

    def _spam(tag: int) -> None:
        barrier.wait()
        for i in range(32):
            log.append(
                api_mod.CompletionRecord(
                    completion_id=f"t{tag}-{i}",
                    backend="byok",
                    ok=True,
                    latency_ms=1.0,
                    at=time.time(),
                    prompt_sha256="p",
                )
            )

    threads = [threading.Thread(target=_spam, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out["ring_threaded_append_exact"] = len(log.all()) == 64 and log.dropped == 64
    return out


def _adversarial_probes() -> dict[str, Any]:
    """Garbage usage payloads must never crash the report or fabricate
    spend; unit-level window and aliasing edges."""
    out: dict[str, Any] = {}
    garbage = {
        "prompt_tokens": "many",  # unbillable — not an int
        "completion_tokens": True,  # a bool is not a count
        "total_tokens": -4,  # provider-reported negative, verbatim
        "cached_tokens": 2.5,  # non-int — dropped
        "note": "x",
    }
    client, _ = _client(
        {
            "byok": lambda: _AnyUsageBackend("g-0", dict(garbage)),
            "local_fx1": lambda: _AnyUsageBackend("n-0", {"prompt_tokens": -3, "total_tokens": -4}),
            "hosted_k3": lambda: _BareBackend(),
        },
        api_key=_ROOT,
    )
    root_h = {"X-API-Key": _ROOT}
    raw, kid = _mint(client, root_h)
    kh = {"X-API-Key": raw}
    _complete(client, "byok", kh)
    _complete(client, "local_fx1", kh, checkpoint_dir="synthetic-ckpt")
    _complete(client, "hosted_k3", kh)
    rep = _usage(client, root_h)
    out["garbage_usage_never_500s"] = rep["records_seen"] == 3
    # verbatim contract: negative ints sum in; non-int/bool values are
    # unbillable and dropped — the report can never NaN or crash
    out["garbage_usage_excluded_from_sums"] = (
        rep["totals"]["prompt_tokens"] == -3
        and rep["totals"]["completion_tokens"] == 0
        and rep["totals"]["total_tokens"] == -8
        and rep["totals"]["other_usage"] == {}
        and rep["totals"]["usage_reported"] == 2
    )
    card = _key_card(client, root_h, kid)
    out["negative_usage_never_bills"] = (
        card["tokens_used"] == 0 and card["served"]["total_tokens"] == -8
    )
    # a backend that never sets last_usage reports nothing
    out["silent_backend_reports_nothing"] = (
        rep["by_backend"]["hosted_k3"]["usage_reported"] == 0
        and rep["by_backend"]["hosted_k3"]["total_tokens"] == 0
        and rep["by_model"]["(none)"]["requests"] == 1
    )
    # a non-dict usage payload is dropped at the wire, not crammed in
    nd_client, _ = _client({"byok": lambda: _AnyUsageBackend("x-0", "not-a-dict")}, api_key=_ROOT)
    _complete(nd_client, "byok", root_h)
    nd = _usage(nd_client, root_h)
    out["nondict_usage_dropped"] = (
        nd["records_seen"] == 1
        and nd["totals"]["usage_reported"] == 0
        and nd["totals"]["total_tokens"] == 0
    )
    # "(none)" is a display label, not a queryable value — filtering on it
    # matches no record (a record's model is None, not the literal)
    out["model_none_label_not_queryable"] = (
        _usage(client, root_h, model="(none)")["records_seen"] == 0
    )
    # unit-level: window edges are inclusive on both sides of `at`
    from fx1.serve.usage_report import aggregate_usage  # noqa: PLC0415

    recs = [
        _FakeRec(backend="b", model="m", ok=True, latency_ms=1.0, at=t)
        for t in (100.0, 200.0, 300.0)
    ]
    out["window_edges_inclusive"] = (
        aggregate_usage(recs, cap=10, dropped=0, since=200.0).records_seen == 2
        and aggregate_usage(recs, cap=10, dropped=0, until=200.0).records_seen == 2
        and aggregate_usage(recs, cap=10, dropped=0, since=200.001).records_seen == 1
        and aggregate_usage(recs, cap=10, dropped=0, until=199.999).records_seen == 1
    )
    # clock skew: a far-future `at` is reported, never clamped to wall time
    future = _FakeRec(backend="b", model="m", ok=True, latency_ms=1.0, at=4e12)
    rep_f = aggregate_usage([*recs, future], cap=10, dropped=0)
    out["future_at_unclamped"] = (
        rep_f.records_seen == 4
        and aggregate_usage([*recs, future], cap=10, dropped=0, since=time.time()).records_seen == 1
    )
    # model aliasing: two backends reporting the same model merge into one
    # bucket; distinct names partition
    a1 = _FakeRec(backend="b1", model="ft:x", ok=True, latency_ms=1.0, at=1.0)
    a2 = _FakeRec(backend="b2", model="ft:x", ok=True, latency_ms=1.0, at=2.0)
    a3 = _FakeRec(backend="b3", model="ckpt-9", ok=True, latency_ms=1.0, at=3.0)
    rep_a = aggregate_usage([a1, a2, a3], cap=10, dropped=0)
    out["model_alias_merges_same_name"] = (
        set(rep_a.by_model) == {"ft:x", "ckpt-9"}
        and rep_a.by_model["ft:x"].requests == 2
        and rep_a.by_model["ckpt-9"].requests == 1
        and set(rep_a.by_backend) == {"b1", "b2", "b3"}
    )
    return out


def _auth_surface_probes() -> dict[str, Any]:
    """Env/loopback credential attribution."""
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))})
    _complete(client, "byok")
    rep = client.get("/harness/usage").json()
    out["loopback_lands_none_bucket"] = (
        rep["by_key"].get("(none)", {}).get("requests") == 1 and rep["totals"]["requests"] == 1
    )
    keyed, _ = _client({"byok": lambda: _MeterBackend("m-0", dict(_BYOK_U))}, api_key=_ROOT)
    _complete(keyed, "byok", {"X-API-Key": _ROOT})
    rep2 = _usage(keyed, {"X-API-Key": _ROOT})
    self_card = keyed.get("/harness/self", headers={"X-API-Key": _ROOT}).json()
    out["env_calls_land_env_bucket"] = (
        rep2["by_key"].get("env", {}).get("requests") == 1
        and rep2["by_key"]["env"]["total_tokens"] == _BYOK_U["total_tokens"]
        and self_card.get("credential") == "env"
        and self_card.get("metered") is False
    )
    return out


def usage_audit() -> dict[str, Any]:
    """Run the accounting-integrity battery; returns literal bools."""
    out: dict[str, Any] = {}
    out.update(_scenario())
    out.update(_refusal_probes())
    out.update(_split_probes())
    out.update(_ring_probes())
    out.update(_concurrency_probes())
    out.update(_adversarial_probes())
    out.update(_auth_surface_probes())
    return out


def usage_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under usage_audit.v1."""
    r = usage_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "usage_audit",
        "schema": "usage_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Usage accounting is conserved: per-backend, per-model, and "
            "per-key buckets each re-derive the totals, and the totals "
            "equal an independent fold over the raw completion records. "
            "Filters partition the retained window disjointly with "
            "inclusive since/until edges on the record's own timestamp; "
            "since>until fails closed 400. Managed-key meters reconcile: "
            "uses counts authenticated requests (budget, window, scope, "
            "and auth refusals never count; a post-auth failure still "
            "does), tokens_used equals the per-record charge formula "
            "summed over the key's records plus the declared batch "
            "usage_total delta (batch items carry no per-item usage — "
            "the shared-backend delta is the honest charge, disclosed by "
            "a zero served-token split). Streaming, /v1/chat n-fanout, "
            "and /v1/messages bill identically to the sync route; "
            "idempotent replays never double-bill. The ring declares its "
            "evictions exactly — totals cover only retained records while "
            "evicted calls stay billed (the charge fires at append; the "
            "ring bounds evidence, not spend, and the key card's "
            "log_dropped marks served as a lower bound). Fallback chains "
            "attribute spend to the serving link. Garbage usage payloads "
            "can never crash or corrupt the report: non-int/bool values "
            "are unbillable and dropped, negatives sum verbatim while "
            "charge_tokens clamps them at zero — a budget never goes "
            "negative."
            if ok
            else f"USAGE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(usage_audit_bench(), indent=2, sort_keys=True))
