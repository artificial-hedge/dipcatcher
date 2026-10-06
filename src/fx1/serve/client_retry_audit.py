"""fx1 serve audit — HarnessClient retry/backoff/timeout semantics.

Lane 183. ``client_audit`` pins the error-map surface and the retry
baseline (which statuses retry, Retry-After honored, writes never
retry unkeyed). This battery extends past it into the mechanics of
the retry loop itself: exact attempt counts per status×idempotency
cell, the sleep schedule as arithmetic (Retry-After vs the doubling
backoff vs ``max_retry_wait_s`` across mixed fault/refusal
sequences), ``Retry-After`` parsing edges (whitespace, scientific,
sign, HTTP-date, non-finite), the ``_last_response_headers`` /
``_last_api_version`` lifecycle across faults vs mapped errors,
circuit-breaker interplay per error class, timeout propagation on
every attempt, and the Idempotency-Key riding every retry verbatim.

All wire probing goes through the scripted-transport seam
(``HarnessClient(transport=...)`` — a fake returning a programmed
sequence of ``(status, headers, body)`` and recording every call);
sleeps ride the injected ``sleep=`` hook, never wall-clock. Measured
bools only; a defect pins ``False`` and is fixed on this PR.
"""

from __future__ import annotations

import contextlib
import threading
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from typing import Any

__all__ = ["client_retry_audit", "client_retry_audit_bench"]

_AUDIT_LOCK = threading.Lock()

_BASE = "https://harness.invalid"  # NOSONAR(S1313) — scripted transport, never dialed
_RA = "Retry-After"
_ERR_BODY = b'{"detail":"refused","code":"refused_test"}'
_ITEMS_BODY = b'{"items":[{"name":"alpha"},{"name":"beta"}]}'
_MSGS: list[dict[str, str]] = [{"role": "user", "content": "ping"}]
_COMP_OK: _Step = (
    200,
    {},
    b'{"backend":"hosted_k3","model":"m","content":"ok",'
    b'"receipt_hashes":[],"replayed":false,"attempts":[],'
    b'"usage":{"tokens_in":1,"tokens_out":1},"completion_id":"c-1"}',
)

_Step = tuple[int, dict[str, str], bytes]


@contextlib.contextmanager
def _audit_context() -> Iterator[None]:
    """Audit lanes are serialized — these probes mutate no shared env
    state but the lock keeps the battery honest alongside siblings."""
    _AUDIT_LOCK.acquire()
    try:
        yield
    finally:
        _AUDIT_LOCK.release()


# --------------------------------------------------------------------------
# scripted transport + recording helpers
# --------------------------------------------------------------------------


def _scripted(
    *steps: _Step | BaseException,
) -> tuple[Any, list[tuple[str, str]], list[dict[str, str]]]:
    """Canned transport: replays ``steps`` — (status, headers, body)
    triples or exceptions to raise — and records (method, path) plus
    the merged request headers per call. Past the script the last step
    repeats (triple or exception), so a "refusal forever" or "fault
    forever" leg is one step."""
    calls: list[tuple[str, str]] = []
    seen_headers: list[dict[str, str]] = []
    it = iter(steps)
    last: _Step | BaseException = (500, {}, b'{"detail":"script exhausted"}')

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,  # NOSONAR(S1172) — contract signature
        headers: dict[str, str],
        timeout_s: float,  # NOSONAR(S1172)
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal last
        calls.append((method, urllib.parse.urlparse(url).path))
        seen_headers.append(dict(headers))
        try:
            step = next(it)
        except StopIteration:
            step = last
        else:
            last = step
        if isinstance(step, BaseException):
            raise step
        return step

    return send, calls, seen_headers


def _mk(transport: Any, **kw: Any) -> Any:
    """A HarnessClient on a scripted/fake transport."""
    from fx1.serve.client import HarnessClient

    return HarnessClient(_BASE, transport=transport, **kw)


def _exc(fn: Callable[[], Any]) -> BaseException | None:
    """The exception ``fn`` raised (None when it returned)."""
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 — measuring the raised class
        return exc
    return None


# --------------------------------------------------------------------------
# probe sections
# --------------------------------------------------------------------------


def _probe_ctor() -> dict[str, bool]:
    """Constructor knob validation — each bound rejects before any
    wire work can happen."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessClient

    tr, _, _ = _scripted()

    def rejects(**kw: Any) -> bool:
        return type(_exc(lambda: HarnessClient(_BASE, transport=tr, **kw))).__name__ == "ValueError"

    out["ctor_rejects_timeout_zero"] = rejects(timeout_s=0)
    out["ctor_rejects_timeout_negative"] = rejects(timeout_s=-1.5)
    out["ctor_rejects_max_retries_negative"] = rejects(max_retries=-1)
    out["ctor_rejects_backoff_zero"] = rejects(retry_backoff_s=0)
    out["ctor_rejects_backoff_negative"] = rejects(retry_backoff_s=-0.25)
    out["ctor_rejects_max_wait_zero"] = rejects(max_retry_wait_s=0)
    out["ctor_rejects_max_wait_negative"] = rejects(max_retry_wait_s=-3)
    out["ctor_accepts_boundary_combo"] = (
        _exc(
            lambda: HarnessClient(
                _BASE,
                transport=tr,
                timeout_s=0.001,
                max_retries=0,
                retry_backoff_s=0.001,
                max_retry_wait_s=0.001,
            )
        )
        is None
    )
    return out


def _probe_attempt_counts() -> dict[str, bool]:
    """``range(retries + 1)`` — max_retries=N means N retries past the
    initial attempt, and only when the call is retryable."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {}, _ITEMS_BODY)
    ra429: _Step = (429, {_RA: "0"}, _ERR_BODY)
    ra503: _Step = (503, {_RA: "0"}, _ERR_BODY)
    fault = HarnessTransportError("dial failed")

    c, calls, _ = _scripted(ra429)
    out["no_retries_means_one_call"] = _exc(_mk(c).commands) is not None and len(calls) == 1

    c, calls, _ = _scripted(ra429)
    cli = _mk(c, max_retries=2, sleep=lambda s: None)
    out["two_retries_means_three_calls"] = _exc(cli.commands) is not None and len(calls) == 3

    c, calls, _ = _scripted(fault)
    cli = _mk(c, max_retries=2, sleep=lambda s: None)
    out["fault_two_retries_three_calls"] = _exc(cli.commands) is not None and len(calls) == 3

    c, calls, _ = _scripted(fault, ra503, ok)
    cli = _mk(c, max_retries=5, sleep=lambda s: None)
    out["mixed_stops_on_first_success"] = cli.commands() == ["alpha", "beta"] and len(calls) == 3

    c, calls, _ = _scripted(fault)
    cli = _mk(c, max_retries=3, sleep=lambda s: None)
    out["write_not_retried_by_default"] = (
        _exc(lambda: cli.complete(_MSGS)) is not None and len(calls) == 1
    )

    c, calls, _ = _scripted(fault)
    cli = _mk(c, max_retries=3, retry_writes=True, sleep=lambda s: None)
    # retry_writes retries transport faults on POST — all 4 attempts
    exc = _exc(lambda: cli.complete(_MSGS))
    out["retry_writes_fault_retries_post"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 4
    )

    c, calls, _ = _scripted(fault, _COMP_OK)
    cli = _mk(c, max_retries=3, sleep=lambda s: None)
    res = cli.complete(_MSGS, idempotency_key="k-1")
    out["keyed_write_fault_retries"] = res.content == "ok" and len(calls) == 2

    # every attempt hits the identical path — no drift mid-retry
    c, calls, _ = _scripted(ra429, ra429, ok)
    cli = _mk(c, max_retries=3, sleep=lambda s: None)
    cli.commands()
    out["retries_hit_same_path"] = calls == [
        ("GET", "/harness/commands"),
        ("GET", "/harness/commands"),
        ("GET", "/harness/commands"),
    ]
    return out


def _probe_sleep_arithmetic() -> dict[str, bool]:
    """The sleep list as arithmetic: the doubling backoff feeds
    transport-fault retries; a Retry-After wait (capped by
    ``max_retry_wait_s``) feeds refusal retries — and each ``backoff
    *= 2`` still runs even when the wait won."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {}, _ITEMS_BODY)
    fault = HarnessTransportError("dial failed")

    def runs(
        *steps: _Step | BaseException, **kw: Any
    ) -> tuple[list[float], list[str], list[tuple[str, str]]]:
        tr, calls, _ = _scripted(*steps)
        sleeps: list[float] = []
        events: list[str] = []

        def send(*a: Any) -> Any:
            events.append("call")
            return tr(*a)

        def sleep(s: float) -> None:
            events.append("sleep")
            sleeps.append(s)

        _exc(_mk(send, sleep=sleep, **kw).commands)
        return sleeps, events, calls

    sleeps, _, _ = runs(fault, fault, fault, ok, max_retries=3, retry_backoff_s=0.1)
    out["backoff_doubles_pure_faults"] = sleeps == [0.1, 0.2, 0.4]

    # RA overrides the grown backoff: fault doubled backoff to 0.2 but
    # the 503's declared wait wins the sleep slot.
    sleeps, _, _ = runs(
        fault, (503, {_RA: "0.5"}, _ERR_BODY), ok, max_retries=3, retry_backoff_s=0.1
    )
    out["ra_overrides_grown_backoff"] = sleeps == [0.1, 0.5]

    # the status branch still doubles backoff for the NEXT fault sleep
    sleeps, _, _ = runs(
        (503, {_RA: "0.5"}, _ERR_BODY), fault, ok, max_retries=3, retry_backoff_s=0.1
    )
    out["backoff_grows_across_status_sleep"] = sleeps == [0.5, 0.2]

    # a declared wait under the cap is slept verbatim — the min() in
    # the loop is dead cover: wait > cap already broke out above
    sleeps, _, _ = runs((429, {_RA: "0.7"}, _ERR_BODY), ok, max_retries=2, max_retry_wait_s=1.0)
    out["ra_under_cap_sleeps_verbatim"] = sleeps == [0.7]

    # boundary: wait == cap retries (the give-up branch is strict >)
    sleeps, _, _ = runs((429, {_RA: "1.0"}, _ERR_BODY), ok, max_retries=2, max_retry_wait_s=1.0)
    out["ra_at_cap_retries"] = sleeps == [1.0]

    sleeps, _, calls = runs(
        (429, {_RA: "1.001"}, _ERR_BODY), ok, max_retries=2, max_retry_wait_s=1.0
    )
    out["ra_over_cap_breaks_no_sleep"] = len(calls) == 1 and sleeps == []

    # ordering: every sleep precedes the retry it belongs to
    sleeps, events, _ = runs(fault, ok, max_retries=1, retry_backoff_s=0.1)
    out["sleep_precedes_retry"] = events == ["call", "sleep", "call"]

    # sleep count tracks retries actually taken
    sleeps, _, calls = runs((429, {_RA: "0"}, _ERR_BODY), ok, max_retries=5)
    out["sleep_count_equals_retries"] = len(sleeps) == len(calls) - 1
    return out


def _probe_retry_after_parsing() -> dict[str, bool]:
    """``_retry_after_s`` is a seconds-only ``float()`` parser with a
    ``max(0, …)`` floor — whitespace/scientific/sign forms ride it, an
    HTTP-date or junk falls to None (not retryable)."""
    out: dict[str, bool] = {}
    ok: _Step = (200, {}, _ITEMS_BODY)

    def sleeps_and_calls(ra_value: str, **kw: Any) -> tuple[list[float], int]:
        tr, calls, _ = _scripted((429, {_RA: ra_value}, _ERR_BODY), ok)
        sleeps: list[float] = []
        cli = _mk(tr, max_retries=2, sleep=lambda s: sleeps.append(s), **kw)
        _exc(cli.commands)
        return sleeps, len(calls)

    s, n = sleeps_and_calls("0")
    out["ra_zero_sleeps_zero"] = s == [0.0] and n == 2
    s, n = sleeps_and_calls("0.25")
    out["ra_fractional"] = s == [0.25] and n == 2
    s, n = sleeps_and_calls(" 3 ")
    out["ra_whitespace_parses"] = s == [3.0] and n == 2
    s, n = sleeps_and_calls("+2")
    out["ra_signed_positive"] = s == [2.0] and n == 2
    s, n = sleeps_and_calls("-5")
    out["ra_negative_clamps_zero"] = s == [0.0] and n == 2
    s, n = sleeps_and_calls("1e1", max_retry_wait_s=5.0)
    out["ra_scientific_over_cap_breaks"] = s == [] and n == 1
    s, n = sleeps_and_calls("Wed, 21 Oct 2015 07:28:00 GMT")
    out["ra_http_date_not_retryable"] = s == [] and n == 1
    s, n = sleeps_and_calls("soon")
    out["ra_junk_not_retryable"] = s == [] and n == 1
    s, n = sleeps_and_calls("")
    out["ra_empty_not_retryable"] = s == [] and n == 1
    s, n = sleeps_and_calls("inf")
    out["ra_infinite_breaks_over_cap"] = s == [] and n == 1
    # ``max(0.0, nan)`` returns 0.0 (``nan > 0`` is False) — a nan wait
    # is treated as "retry immediately", the opposite edge of inf
    s, n = sleeps_and_calls("nan")
    out["ra_nan_clamps_zero_retries"] = s == [0.0] and n == 2

    # case-insensitive header name — transports differ in casing
    for name in ("RETRY-AFTER", "retry-after", "Retry-After"):
        tr, calls, _ = _scripted((429, {name: "0"}, _ERR_BODY), ok)
        cli = _mk(tr, max_retries=1, sleep=lambda x: None)
        _exc(cli.commands)
        out[f"ra_header_case_{name.replace('-', '_').lower()}"] = len(calls) == 2
    return out


def _probe_error_lifecycle() -> dict[str, bool]:
    """What escapes, and what the shared slots hold afterwards:
    mapped non-transport errors keep the response's headers; anything
    shaped ``HarnessTransportError`` (transport fault OR mapped 5xx)
    clears them and trips the circuit counter."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {"X-Fx1-Api-Version": "2025.1"}, _ITEMS_BODY)
    s500: _Step = (500, {"X-Err": "1"}, b"plain crash")
    s503: _Step = (503, {"X-Err": "2"}, b'{"detail":"busy","code":"over_capacity"}')
    s429: _Step = (429, {"X-Err": "3"}, _ERR_BODY)
    s404: _Step = (404, {"X-Err": "4"}, b'{"detail":"nope","code":"missing"}')
    fault = HarnessTransportError("dial failed")

    # mapped 500 is a HarnessTransportError → headers cleared on escape
    tr, _, _ = _scripted(s500)
    cli = _mk(tr)
    exc = _exc(cli.commands)
    out["mapped_500_is_transport_error"] = type(exc).__name__ == "HarnessTransportError"
    out["mapped_500_clears_headers"] = cli._last_response_headers == {}

    # mapped 503/404 are non-transport classes → response headers retained
    tr, _, _ = _scripted(s503)
    cli = _mk(tr)
    exc = _exc(cli.commands)
    out["mapped_503_not_transport"] = type(exc).__name__ == "BackendNotConfiguredError"
    out["mapped_503_keeps_headers"] = cli._last_response_headers.get("x-err") == "2"
    tr, _, _ = _scripted(s404)
    cli = _mk(tr)
    exc = _exc(cli.commands)
    out["mapped_404_is_keyerror"] = type(exc).__name__ == "KeyError"
    out["mapped_404_keeps_headers"] = cli._last_response_headers.get("x-err") == "4"
    tr, _, _ = _scripted(s429)
    cli = _mk(tr)
    exc = _exc(cli.commands)
    out["mapped_429_is_transport"] = type(exc).__name__ == "HarnessTransportError"
    out["mapped_429_clears_headers"] = cli._last_response_headers == {}

    # success stores lower-cased headers and the api version
    tr, _, _ = _scripted(ok)
    cli = _mk(tr)
    cli.commands()
    out["success_headers_lowercased"] = cli._last_response_headers.get("x-fx1-api-version") == (
        "2025.1"
    )
    out["success_tracks_api_version"] = cli._last_api_version == "2025.1"

    # a transport fault clears headers but leaves the stale api version
    tr, _, _ = _scripted(ok, fault)
    cli = _mk(tr)
    cli.commands()
    _exc(cli.commands)
    out["fault_clears_headers"] = cli._last_response_headers == {}
    out["fault_keeps_api_version"] = cli._last_api_version == "2025.1"

    # a NON-transport transport raise propagates raw: no retry, no
    # header clear, no sleep — the loop only handles its own class
    tr, calls, _ = _scripted(ValueError("weird transport"))
    sleeps: list[float] = []
    cli = _mk(tr, max_retries=3, sleep=lambda s: sleeps.append(s))
    exc = _exc(cli.commands)
    out["foreign_exc_propagates_unretried"] = (
        type(exc).__name__ == "ValueError" and len(calls) == 1 and sleeps == []
    )

    # exhaustion naming: the long-wait break produces the literal
    # "exhausted" message; a final-attempt refusal maps normally
    tr, calls, _ = _scripted((429, {_RA: "9"}, _ERR_BODY), ok)
    cli = _mk(tr, max_retries=3, max_retry_wait_s=1.0)
    exc = _exc(cli.commands)
    out["long_wait_break_message"] = (
        type(exc).__name__ == "HarnessTransportError"
        and str(exc) == "harness GET /harness/commands exhausted 3 retries"
        and len(calls) == 1
    )
    # a long wait only breaks mid-loop — on the FINAL attempt the
    # ``attempt < retries`` guard skips the retry branch entirely and
    # the refusal maps through the normal table
    short503: _Step = (503, {_RA: "0"}, _ERR_BODY)
    tr, calls, _ = _scripted(short503, short503, (503, {_RA: "9"}, _ERR_BODY))
    cli = _mk(tr, max_retries=2, max_retry_wait_s=1.0, sleep=lambda s: None)
    exc = _exc(cli.commands)
    out["final_attempt_long_wait_maps_normally"] = (
        type(exc).__name__ == "BackendNotConfiguredError" and len(calls) == 3
    )
    return out


def _probe_circuit_interplay() -> dict[str, bool]:
    """Only ``HarnessTransportError``-class escapes trip the counter —
    mapped refusals typed otherwise (503→BackendNotConfiguredError,
    401→auth) never do, so a service that refuses honestly stays
    reachable while one that faults reliably gets shed."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {}, _ITEMS_BODY)
    fault = HarnessTransportError("dial failed")
    ra503: _Step = (503, {_RA: "0"}, b'{"detail":"busy","code":"over_capacity"}')

    # an exhausted 503+RA maps to BackendNotConfiguredError → no trip
    tr, calls, _ = _scripted(ra503, ra503, ok)
    clock_t = [0.0]
    cli = _mk(
        tr,
        max_retries=1,
        sleep=lambda s: None,
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: clock_t[0],
    )
    exc = _exc(cli.commands)
    out["mapped_503_exhaustion_skips_circuit"] = (
        type(exc).__name__ == "BackendNotConfiguredError"
        and cli.commands() == ["alpha", "beta"]
        and len(calls) == 3
    )

    # threshold=1: a single fault opens; fail-fast adds no transport call
    tr, calls, _ = _scripted(fault)
    cli = _mk(
        tr,
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: clock_t[0],
        sleep=lambda s: None,
    )
    _exc(cli.commands)
    exc = _exc(cli.commands)
    out["threshold_one_opens_on_first_fault"] = (
        type(exc).__name__ == "HarnessTransportError"
        and "circuit open" in str(exc)
        and len(calls) == 1
    )

    # past reset, the next call is a real probe: success closes the gate
    clock_t[0] += 31.0
    tr2, calls2, _ = _scripted(ok)
    cli._transport = tr2  # noqa: SLF001 — audit probes the seams directly
    out["half_open_probe_success_closes"] = cli.commands() == ["alpha", "beta"] and len(calls2) == 1
    # closed again: another call goes through
    out["closed_circuit_passes"] = cli.commands() == ["alpha", "beta"] and len(calls2) == 2

    # a half-open fault re-opens the window — boundary at open_until
    tr3, calls3, _ = _scripted(fault)
    cli._transport = tr3  # noqa: SLF001
    clock_t[0] += 1.0
    _exc(cli.commands)  # probe faults → trip → open at clock+30
    exc = _exc(cli.commands)
    out["half_open_fault_reopens"] = "circuit open" in str(exc) and len(calls3) == 1

    # open-until boundary is strict <: exactly at the deadline the gate opens
    clock_t[0] += 30.0
    tr4, calls4, _ = _scripted(ok)
    cli._transport = tr4  # noqa: SLF001
    out["open_until_boundary_is_exclusive"] = (
        cli.commands() == ["alpha", "beta"] and len(calls4) == 1
    )

    # fail-fast sleep-free: no sleep recorded while the gate is shut
    tr, calls, _ = _scripted(fault)
    sleeps: list[float] = []
    cli = _mk(
        tr,
        max_retries=0,
        sleep=lambda s: sleeps.append(s),
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: clock_t[0],
    )
    _exc(cli.commands)
    _exc(cli.commands)
    out["open_circuit_never_sleeps"] = sleeps == []

    # consecutive counting: a success between faults holds the gate open
    tr, calls, _ = _scripted(fault, ok, fault, fault)
    cli = _mk(
        tr,
        sleep=lambda s: None,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
        clock=lambda: clock_t[0],
    )
    _exc(cli.commands)
    cli.commands()
    _exc(cli.commands)
    _exc(cli.commands)
    exc = _exc(cli.commands)
    out["counter_needs_consecutive_faults"] = "circuit open" in str(exc) and len(calls) == 4
    return out


def _probe_timeout_and_key_on_wire() -> dict[str, bool]:
    """``timeout_s`` reaches the transport verbatim on every attempt;
    the Idempotency-Key rides each retry unchanged (regenerating it
    per attempt would defeat server-side dedup)."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {}, _ITEMS_BODY)
    fault = HarnessTransportError("dial failed")

    timeouts: list[float] = []
    tr, calls, hdrs = _scripted(fault, fault, ok)
    orig = tr

    def timed(*a: Any) -> Any:
        timeouts.append(a[4])
        return orig(*a)

    cli = _mk(timed, max_retries=2, timeout_s=7.5, sleep=lambda s: None)
    cli.commands()
    out["timeout_verbatim_each_attempt"] = timeouts == [7.5, 7.5, 7.5] and len(calls) == 3

    ra429: _Step = (429, {_RA: "0"}, _ERR_BODY)
    tr, calls, hdrs = _scripted(ra429, _COMP_OK)
    cli = _mk(tr, max_retries=2, sleep=lambda s: None)
    cli.complete(_MSGS, idempotency_key="k-77")
    keys = [h.get("Idempotency-Key") for h in hdrs]
    out["idempotency_key_on_every_attempt"] = keys == ["k-77", "k-77"] and len(calls) == 2

    tr, calls, hdrs = _scripted(_COMP_OK)
    cli = _mk(tr, max_retries=2, sleep=lambda s: None)
    cli.complete(_MSGS)
    out["unkeyed_write_sends_no_key"] = hdrs[0].get("Idempotency-Key") is None

    # api_key auth header rides every attempt too
    tr, calls, hdrs = _scripted(fault, ok)
    cli = _mk(tr, api_key="k3y-material", max_retries=1, sleep=lambda s: None)
    cli.commands()
    out["api_key_header_every_attempt"] = [h.get("X-API-Key") for h in hdrs] == [
        "k3y-material",
        "k3y-material",
    ]
    return out


def _probe_retry_under_parallelism() -> dict[str, bool]:
    """Parallel callers share the client: the retry policy is per-call
    (sleep/backoff are locals), but the circuit counter and the
    last-headers slot are shared — pin what's actually coherent."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError

    ok: _Step = (200, {}, _ITEMS_BODY)

    # a thread-safe script: N callers, each call gets its own step by
    # index — refusal-then-ok per caller, all under a Barrier start
    n = 8
    steps = [step for _ in range(n) for step in ((429, {_RA: "0"}, _ERR_BODY), ok)]
    lock = threading.Lock()
    it = iter(steps)
    calls: list[tuple[str, str]] = []

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        calls.append((method, url))
        with lock:
            return next(it)

    sleeps: list[float] = []
    sleep_lock = threading.Lock()

    def sleep(s: float) -> None:
        with sleep_lock:
            sleeps.append(s)

    cli = _mk(send, max_retries=1, sleep=sleep)
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        e = _exc(cli.commands)
        if e is not None:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    out["parallel_retries_all_succeed"] = not errors and len(calls) == 2 * n
    out["parallel_sleep_count_matches"] = len(sleeps) == n

    # circuit under parallel faults: after enough HarnessTransportError
    # escapes the gate is observably open — some later call fails fast
    # without reaching the transport
    calls2: list[int] = []

    def send2(*a: Any) -> Any:
        calls2.append(1)
        raise HarnessTransportError("dial failed")

    cli2 = _mk(
        send2,
        circuit_breaker_threshold=4,
        circuit_reset_s=30.0,
        sleep=lambda s: None,
    )
    n2 = 24
    barrier2 = threading.Barrier(n2)
    failfast: list[str] = []

    def worker2() -> None:
        barrier2.wait()
        e = _exc(cli2.commands)
        if e is not None and "circuit open" in str(e):
            failfast.append("open")

    threads2 = [threading.Thread(target=worker2) for _ in range(n2)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join(timeout=60)
    out["parallel_faults_eventually_open_circuit"] = len(failfast) >= 1 and len(calls2) < n2
    return out


# --------------------------------------------------------------------------
# battery + bench
# --------------------------------------------------------------------------


def client_retry_audit() -> dict[str, bool]:
    """The full client-retry battery. Pure scripted transport — no
    sockets, no wall-clock sleeps; safe under ``pytest -n``."""
    with _audit_context():
        results: dict[str, bool] = {}
        results.update(_probe_ctor())
        results.update(_probe_attempt_counts())
        results.update(_probe_sleep_arithmetic())
        results.update(_probe_retry_after_parsing())
        results.update(_probe_error_lifecycle())
        results.update(_probe_circuit_interplay())
        results.update(_probe_timeout_and_key_on_wire())
        results.update(_probe_retry_under_parallelism())
        return results


def client_retry_audit_bench(results: dict[str, bool] | None = None) -> dict[str, Any]:
    """Seal client-retry results; run the battery when omitted."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = client_retry_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "client_retry_audit",
        "schema": "client_retry_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "scripted in-process transport (no sockets)",
            "not_verified": [
                "real socket timeouts (urllib deadline vs injected timeout_s)",
                "Retry-After HTTP-date correctness (parser is seconds-only by design)",
                "server-side dedup of the Idempotency-Key (client pins the wire value only)",
                "retry semantics inside SSE stream legs (separate surface)",
            ],
        },
        "interpretation": (
            "HarnessClient retry holds: attempt counts are retries+1 on "
            "retryable calls only (unkeyed writes never retry), the "
            "doubling backoff feeds transport-fault sleeps while "
            "Retry-After — capped by max_retry_wait_s and parsed as "
            "seconds only — feeds refusal sleeps, a declared wait past "
            "the budget breaks without sleeping and names the policy in "
            "the exhausted message, response headers clear only on "
            "transport-class escapes while mapped refusals retain them, "
            "the circuit trips on transport-class faults alone and "
            "re-arms strictly after its reset window, timeout_s and the "
            "Idempotency-Key reach every attempt verbatim, and parallel "
            "callers each get their own retry schedule over the shared "
            "circuit counter. SYNTHETIC scripted transport only — no "
            "research claim."
            if ok
            else f"HARNESS CLIENT RETRY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(client_retry_audit_bench(), indent=1))
