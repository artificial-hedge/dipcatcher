"""Guidance, retries, and identity probes for hosted model endpoints."""

from __future__ import annotations

import io
import json
import random
import socket
import threading
import urllib.error
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from fx1.interactive.endpoint import (
    FailureClass,
    Guidance,
    IdentityReport,
    OperationCancelled,
    RetryPolicy,
    call_with_retries,
    explain_backend_error,
    format_guidance,
    probe_identity,
)

SECRET_TOKEN = "sk-ABCDEFGHIJKLMNOP"
SECRET_URL = "https://user:pw@host/x"
LEAK_URL = "https://user:pw@api.example.test/v1/chat/completions"


class _ProbeCancelled(Exception):
    """Name contains Cancelled so the classifier treats it as a cancel."""


class _Backend:
    def __init__(self, payload: object | None = None, exc: BaseException | None = None) -> None:
        self.payload = payload
        self.exc = exc

    def identity_probe(self) -> object:
        if self.exc is not None:
            raise self.exc
        return self.payload


def _fixed_now() -> datetime:
    return datetime(2026, 10, 12, 2, 47, tzinfo=UTC)


def _probe(
    backend: _Backend,
    *,
    model: str = "fx1",
    host: str = "api.example.test",
    ticks: tuple[float, float] = (1.0, 1.5),
) -> IdentityReport:
    clock = iter(ticks)

    def _clock() -> float:
        return next(clock)

    return probe_identity(
        backend,
        model=model,
        endpoint_host=host,
        clock=_clock,
        now=_fixed_now,
    )


@pytest.mark.parametrize(
    ("exc", "failure", "retryable", "http_status"),
    [
        (RuntimeError("hosted_k3 endpoint returned HTTP 401"), FailureClass.AUTH, False, 401),
        (RuntimeError("hosted_k3 endpoint returned HTTP 403"), FailureClass.AUTH, False, 403),
        (RuntimeError("hosted_k3 endpoint returned HTTP 404"), FailureClass.NOT_FOUND, False, 404),
        (RuntimeError("hosted_k3 endpoint returned HTTP 400"), FailureClass.MALFORMED, False, 400),
        (RuntimeError("hosted_k3 endpoint returned HTTP 422"), FailureClass.MALFORMED, False, 422),
        (
            RuntimeError("hosted_k3 endpoint returned HTTP 408"),
            FailureClass.RATE_LIMITED,
            True,
            408,
        ),
        (
            RuntimeError("hosted_k3 endpoint returned HTTP 429"),
            FailureClass.RATE_LIMITED,
            True,
            429,
        ),
        (RuntimeError("hosted_k3 endpoint returned HTTP 500"), FailureClass.SERVER, True, 500),
        (RuntimeError("hosted_k3 endpoint returned HTTP 503"), FailureClass.SERVER, True, 503),
        (RuntimeError("hosted_k3 endpoint returned HTTP 599"), FailureClass.SERVER, True, 599),
        (
            RuntimeError("hosted_k3 endpoint unreachable (gaierror)"),
            FailureClass.UNREACHABLE,
            True,
            None,
        ),
        (
            RuntimeError("hosted_k3 endpoint unreachable (TimeoutError)"),
            FailureClass.UNREACHABLE,
            True,
            None,
        ),
        (
            RuntimeError("hosted_k3 endpoint transport fault (TimeoutError)"),
            FailureClass.TIMEOUT,
            True,
            None,
        ),
        (
            RuntimeError("hosted_k3 endpoint transport fault (TIMED OUT)"),
            FailureClass.TIMEOUT,
            True,
            None,
        ),
        (
            RuntimeError("hosted_k3 endpoint transport fault (ConnectionResetError)"),
            FailureClass.UNREACHABLE,
            True,
            None,
        ),
        (
            RuntimeError("hosted_k3 endpoint transport fault (HTTPException)"),
            FailureClass.UNREACHABLE,
            True,
            None,
        ),
        (TimeoutError(), FailureClass.TIMEOUT, True, None),
        (TimeoutError("read timed out"), FailureClass.TIMEOUT, True, None),
        (TimeoutError("unreachable host"), FailureClass.TIMEOUT, True, None),
        (
            RuntimeError("malformed hosted_k3 completion payload: not JSON"),
            FailureClass.MALFORMED,
            False,
            None,
        ),
        (
            RuntimeError("malformed completion payload: content is dict, not str"),
            FailureClass.MALFORMED,
            False,
            None,
        ),
        (
            RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials"),
            FailureClass.UNCONFIGURED,
            False,
            None,
        ),
        (
            RuntimeError("no endpoint for model 'fx1'; run dipcatcher"),
            FailureClass.UNCONFIGURED,
            False,
            None,
        ),
        (RuntimeError("BYOK backend is not configured"), FailureClass.UNCONFIGURED, False, None),
        (RuntimeError("FX1_BASE_URL must be an HTTPS URL"), FailureClass.BAD_URL, False, None),
        (
            RuntimeError("FX1_BASE_URL must be a well-formed HTTPS URL"),
            FailureClass.BAD_URL,
            False,
            None,
        ),
        (
            ValueError("invalid stored base URL: must be an HTTPS URL"),
            FailureClass.BAD_URL,
            False,
            None,
        ),
        (
            ValueError("invalid FX1_BASE_URL: port must be 1-65535"),
            FailureClass.BAD_URL,
            False,
            None,
        ),
        (KeyboardInterrupt(), FailureClass.CANCELLED, False, None),
        (_ProbeCancelled("stop"), FailureClass.CANCELLED, False, None),
        (
            _ProbeCancelled("hosted_k3 endpoint returned HTTP 500"),
            FailureClass.CANCELLED,
            False,
            None,
        ),
        (RuntimeError("hosted_k3 endpoint returned HTTP 409"), FailureClass.UNKNOWN, False, 409),
        (RuntimeError("socket blew up"), FailureClass.UNKNOWN, False, None),
    ],
)
def test_explain_backend_error_table(
    exc: BaseException, failure: FailureClass, retryable: bool, http_status: int | None
) -> None:
    guidance = explain_backend_error(exc, model="fx1")
    assert guidance.failure is failure
    assert guidance.retryable is retryable
    assert guidance.http_status == http_status


def test_specified_actions_and_rejected_request_summary() -> None:
    auth = explain_backend_error(RuntimeError("HTTP 401"), model="fx1")
    assert auth.action == "check the API key for fx1: /keys set fx1"
    forbidden = explain_backend_error(RuntimeError("HTTP 403"), model="fx1-lite")
    assert forbidden.action == "check the API key for fx1-lite: /keys set fx1-lite"
    missing = explain_backend_error(RuntimeError("HTTP 404"), model="fx1")
    assert missing.action == (
        "the endpoint URL must be the full chat-completions route "
        "(e.g. https://<host>/v1/chat/completions); fix it with /keys set fx1"
    )
    rejected = explain_backend_error(RuntimeError("HTTP 400"), model="fx1")
    assert rejected.summary == (
        "request rejected; the model id 'fx1' may be unknown to this service or the payload invalid"
    )
    assert (
        explain_backend_error(RuntimeError("HTTP 422"), model="fx1-lite").summary
        == "request rejected; the model id 'fx1-lite' may be unknown "
        "to this service or the payload invalid"
    )
    assert (
        "request rejected"
        not in explain_backend_error(
            RuntimeError("malformed hosted_k3 completion payload: not JSON")
        ).summary
    )
    limited = explain_backend_error(RuntimeError("HTTP 429"))
    assert limited.action == "wait and retry; reduce concurrent runs"
    unset = explain_backend_error(RuntimeError("MOONSHOT_API_KEY is not set; missing"))
    assert unset.action == "/keys set fx1"
    bad = explain_backend_error(ValueError("invalid stored base URL: must be an HTTPS URL"))
    assert bad.action == "/keys set fx1 with an https:// URL"


def test_classified_guidance_drops_key_and_userinfo() -> None:
    exc = RuntimeError(f"HTTP 401 for key {SECRET_TOKEN} at {SECRET_URL}")
    guidance = explain_backend_error(exc, model="fx1")
    rendered = guidance.summary + guidance.action + format_guidance(guidance)
    assert SECRET_TOKEN not in rendered
    assert "user:pw" not in rendered
    assert SECRET_URL not in rendered
    assert guidance.failure is FailureClass.AUTH


def test_unknown_summary_keeps_hostname_only() -> None:
    blob = "C" * 32
    kept = "B" * 31
    exc = RuntimeError(
        f"boom Bearer supersecret {SECRET_TOKEN} key-other {LEAK_URL} id {blob} keep {kept}"
    )
    guidance = explain_backend_error(exc)
    assert guidance.failure is FailureClass.UNKNOWN
    assert guidance.retryable is False
    assert guidance.summary.startswith("RuntimeError: ")
    for secret in (SECRET_TOKEN, "user:pw", "supersecret", "key-other", blob, LEAK_URL, "/v1/chat"):
        assert secret not in guidance.summary
        assert secret not in guidance.action
        assert secret not in format_guidance(guidance)
    assert "api.example.test" in guidance.summary
    assert kept in guidance.summary
    assert "https://" not in guidance.summary


def test_unknown_summary_is_class_name_when_message_is_only_a_secret() -> None:
    guidance = explain_backend_error(RuntimeError(SECRET_TOKEN))
    assert guidance.summary == "RuntimeError"
    assert SECRET_TOKEN not in format_guidance(guidance)


def test_format_guidance_variants() -> None:
    auth = explain_backend_error(RuntimeError("hosted_k3 endpoint returned HTTP 401"))
    assert format_guidance(auth) == (
        "model endpoint error (auth, HTTP 401): authentication failed "
        "→ fix: check the API key for fx1: /keys set fx1"
    )
    limited = explain_backend_error(RuntimeError("hosted_k3 endpoint returned HTTP 429"))
    assert format_guidance(limited) == (
        "model endpoint error (rate_limited (retryable), HTTP 429): "
        "the endpoint asked the client to retry later "
        "→ fix: wait and retry; reduce concurrent runs"
    )
    down = explain_backend_error(RuntimeError("hosted_k3 endpoint unreachable (gaierror)"))
    assert format_guidance(down) == (
        "model endpoint error (unreachable (retryable)): "
        "the endpoint is unreachable (DNS or connection) "
        "→ fix: check the network and the endpoint host"
    )
    assert "HTTP" not in format_guidance(down)
    manual = Guidance(
        failure=FailureClass.CANCELLED,
        summary="the operation was cancelled",
        action="rerun the command",
        retryable=False,
        http_status=None,
    )
    assert format_guidance(manual) == (
        "model endpoint error (cancelled): the operation was cancelled → fix: rerun the command"
    )


def test_retry_policy_validate_accepts_defaults_and_bounds() -> None:
    RetryPolicy().validate()
    RetryPolicy(max_retries=0, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0).validate()
    RetryPolicy(jitter=1.0, base_delay_s=1, max_delay_s=1).validate()


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        (RetryPolicy(max_retries=-1), "max_retries"),
        (RetryPolicy(max_retries=True), "max_retries"),  # type: ignore[arg-type]
        (RetryPolicy(base_delay_s=-0.1), "base_delay_s"),
        (RetryPolicy(max_delay_s=-1), "max_delay_s"),
        (RetryPolicy(base_delay_s=float("nan")), "base_delay_s"),
        (RetryPolicy(jitter=float("inf")), "jitter"),
        (RetryPolicy(max_delay_s=0.1, base_delay_s=0.5), "max_delay_s"),
        (RetryPolicy(jitter=1.01), "jitter"),
        (RetryPolicy(jitter=-0.1), "jitter"),
    ],
)
def test_retry_policy_validate_rejects(policy: RetryPolicy, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        policy.validate()


def _expected_delays(policy: RetryPolicy, retries: int, seed: int) -> list[float]:
    generator = random.Random(seed)
    delays: list[float] = []
    for attempt in range(retries):
        capped = min(policy.max_delay_s, policy.base_delay_s * (2.0**attempt))
        delays.append(capped * (1.0 + policy.jitter * generator.uniform(-1.0, 1.0)))
    return delays


def test_call_with_retries_succeeds_on_third_attempt() -> None:
    calls = {"n": 0}
    delays: list[float] = []
    seen: list[tuple[int, Guidance, float]] = []
    policy = RetryPolicy(max_retries=2, base_delay_s=0.5, max_delay_s=8.0, jitter=0.1)

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("hosted_k3 endpoint returned HTTP 503")
        return "ok"

    def on_retry(attempt_number: int, guidance: Guidance, delay: float) -> None:
        seen.append((attempt_number, guidance, delay))

    result = call_with_retries(
        fn,
        policy=policy,
        sleep=delays.append,
        on_retry=on_retry,
        rng=random.Random(0),
        model="fx1",
    )
    expected = _expected_delays(policy, 2, seed=0)
    assert result == "ok"
    assert calls["n"] == 3
    assert delays == expected
    assert [item[0] for item in seen] == [1, 2]
    assert seen[0][1].failure is FailureClass.SERVER
    assert seen[0][1].retryable is True
    assert seen[0][2] == delays[0]
    for attempt, delay in enumerate(delays):
        capped = min(policy.max_delay_s, policy.base_delay_s * (2.0**attempt))
        assert (
            capped * (1.0 - policy.jitter) - 1e-12
            <= delay
            <= capped * (1.0 + policy.jitter) + 1e-12
        )


def test_call_with_retries_caps_delay() -> None:
    calls = {"n": 0}
    delays: list[float] = []
    policy = RetryPolicy(max_retries=3, base_delay_s=5.0, max_delay_s=8.0, jitter=0.0)

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 4:
            raise TimeoutError("timed out")
        return "ok"

    assert (
        call_with_retries(
            fn,
            policy=policy,
            sleep=delays.append,
            rng=random.Random(1),
        )
        == "ok"
    )
    assert calls["n"] == 4
    assert delays == [5.0, 8.0, 8.0]


def test_call_with_retries_does_not_retry_auth() -> None:
    calls = {"n": 0}
    exc = RuntimeError("hosted_k3 endpoint returned HTTP 401")

    def fn() -> None:
        calls["n"] += 1
        raise exc

    def on_retry(attempt_number: int, guidance: Guidance, delay: float) -> None:
        raise AssertionError(attempt_number)

    with pytest.raises(RuntimeError) as caught:
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=2),
            sleep=lambda _delay: pytest.fail("slept"),
            on_retry=on_retry,
            rng=random.Random(0),
        )
    assert caught.value is exc
    assert calls["n"] == 1


def test_call_with_retries_reraises_last_exception_when_exhausted() -> None:
    calls = {"n": 0}
    last = RuntimeError("hosted_k3 endpoint returned HTTP 500")

    def fn() -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("hosted_k3 endpoint returned HTTP 503")
        raise last

    with pytest.raises(RuntimeError) as caught:
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=2, jitter=0.0),
            sleep=lambda _delay: None,
            rng=random.Random(0),
        )
    assert caught.value is last
    assert calls["n"] == 3


def test_call_with_retries_cancel_in_on_retry() -> None:
    cancel = threading.Event()
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise RuntimeError("hosted_k3 endpoint returned HTTP 503")

    def on_retry(attempt_number: int, guidance: Guidance, delay: float) -> None:
        assert attempt_number == 1
        assert isinstance(guidance, Guidance)
        assert guidance.retryable is True
        assert delay > 0
        cancel.set()

    with pytest.raises(OperationCancelled, match="cancelled before retry 1"):
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=2, base_delay_s=0.5, jitter=0.0),
            cancel_event=cancel,
            sleep=lambda _delay: pytest.fail("slept"),
            on_retry=on_retry,
            rng=random.Random(0),
        )
    assert calls["n"] == 1


def test_call_with_retries_cancel_after_sleep() -> None:
    cancel = threading.Event()
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise RuntimeError("hosted_k3 endpoint returned HTTP 503")

    def sleeper(_delay: float) -> None:
        cancel.set()

    with pytest.raises(OperationCancelled, match="cancelled before retry 1"):
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=2, jitter=0.0),
            cancel_event=cancel,
            sleep=sleeper,
            rng=random.Random(0),
        )
    assert calls["n"] == 1


def test_keyboard_interrupt_propagates_immediately() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=4),
            sleep=lambda _delay: pytest.fail("slept"),
            rng=random.Random(0),
        )
    assert calls["n"] == 1


def test_on_retry_receives_attempt_guidance_and_delay() -> None:
    captured: dict[str, object] = {}

    def fn() -> str:
        if "seen" not in captured:
            captured["seen"] = True
            raise TimeoutError("timed out")
        return "ok"

    def on_retry(attempt_number: int, guidance: Guidance, delay: float) -> None:
        captured["args"] = (attempt_number, guidance, delay)

    result = call_with_retries(
        fn,
        policy=RetryPolicy(max_retries=1, base_delay_s=0.5, max_delay_s=8.0, jitter=0.0),
        sleep=lambda _delay: None,
        on_retry=on_retry,
        rng=random.Random(0),
        model="fx1-lite",
    )
    attempt_number, guidance, delay = captured["args"]
    assert result == "ok"
    assert attempt_number == 1
    assert isinstance(guidance, Guidance)
    assert guidance.failure is FailureClass.TIMEOUT
    assert guidance.retryable is True
    assert delay == pytest.approx(0.5)


def test_cancelled_exception_is_not_retried() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise _ProbeCancelled("hosted_k3 endpoint returned HTTP 500")

    with pytest.raises(_ProbeCancelled):
        call_with_retries(
            fn,
            policy=RetryPolicy(max_retries=3),
            sleep=lambda _delay: pytest.fail("slept"),
            rng=random.Random(0),
        )
    assert calls["n"] == 1


def test_call_with_retries_validates_policy_before_calling() -> None:
    def fn() -> str:
        raise AssertionError("called")

    with pytest.raises(ValueError, match="max_retries"):
        call_with_retries(fn, policy=RetryPolicy(max_retries=-1))


def test_call_with_retries_returns_without_sleep_on_first_success() -> None:
    assert call_with_retries(lambda: 7) == 7


def test_probe_identity_match_uses_injected_clock() -> None:
    start, end = 1.0, 1.5
    report = _probe(_Backend({"model": "fx1", "id": "cmpl-1"}), ticks=(start, end))
    assert report.ok is True
    assert report.served_model == "fx1"
    assert report.identity_matches is True
    assert report.failure is None
    assert report.http_status is None
    assert report.latency_ms == (end - start) * 1000.0
    assert report.checked_at == "2026-10-12T02:47:00+00:00"
    assert report.endpoint_host == "api.example.test"
    assert report.render() == "fx1 @ api.example.test: OK — served model 'fx1' in 500 ms"


def test_probe_identity_prefix_matches_without_a_note() -> None:
    report = _probe(_Backend({"model": "fx1-0905"}))
    assert report.ok is True
    assert report.identity_matches is True
    assert "note:" not in report.render()


def test_probe_identity_mismatch_records_a_note() -> None:
    report = _probe(
        _Backend({"model": "kimi-k3-0905"}),
        host="api.moonshot.ai",
        ticks=(0.0, 0.812),
    )
    assert report.ok is True
    assert report.identity_matches is False
    rendered = report.render()
    assert rendered.startswith("fx1 @ api.moonshot.ai: OK — served model 'kimi-k3-0905' in ")
    assert rendered.endswith(
        "\n  note: served identity differs from the requested model id "
        "(provider routing); recorded, not an error"
    )
    assert "FAILED" not in rendered


def test_probe_identity_auth_failure_renders_fix() -> None:
    report = _probe(
        _Backend(exc=RuntimeError("hosted_k3 endpoint returned HTTP 401")),
        host="api.moonshot.ai",
        ticks=(5.0, 5.04),
    )
    assert report.ok is False
    assert report.failure is FailureClass.AUTH
    assert report.http_status == 401
    assert report.served_model is None
    assert report.identity_matches is None
    assert report.latency_ms == pytest.approx(40.0)
    rendered = report.render()
    assert "FAILED (auth, HTTP 401)" in rendered
    assert "/keys set fx1" in rendered
    assert rendered == (
        "fx1 @ api.moonshot.ai: FAILED (auth, HTTP 401) — authentication failed "
        "→ fix: check the API key for fx1: /keys set fx1"
    )


def test_probe_identity_omits_unreported_model_id() -> None:
    report = _probe(_Backend({"id": "cmpl-1", "usage": {"total_tokens": 1}}))
    assert report.ok is True
    assert report.served_model is None
    assert report.identity_matches is None
    assert report.render() == "fx1 @ api.example.test: OK — provider did not report a model id"


def test_probe_identity_non_dict_is_malformed_and_does_not_raise() -> None:
    report = _probe(_Backend(["nope"]))
    assert report.ok is False
    assert report.failure is FailureClass.MALFORMED
    assert report.http_status is None


def test_probe_identity_redacts_unknown_exception_text() -> None:
    report = _probe(_Backend(exc=RuntimeError(f"socket blew up {SECRET_TOKEN} {LEAK_URL}")))
    assert report.ok is False
    assert report.failure is FailureClass.UNKNOWN
    rendered = report.detail + report.render()
    assert SECRET_TOKEN not in rendered
    assert "user:pw" not in rendered
    assert LEAK_URL not in rendered


def test_probe_identity_converts_non_utc_now_to_utc_seconds() -> None:
    report = probe_identity(
        _Backend({"model": "fx1"}),
        model="fx1",
        endpoint_host="h",
        clock=lambda: 0.0,
        now=lambda: datetime(2026, 10, 12, 4, 47, tzinfo=timezone(timedelta(hours=2))),
    )
    assert report.checked_at == "2026-10-12T02:47:00+00:00"


def test_probe_server_failure_marks_retryable() -> None:
    report = _probe(_Backend(exc=RuntimeError("hosted_k3 endpoint returned HTTP 503")))
    rendered = report.render()
    assert "FAILED (server (retryable), HTTP 503)" in rendered
    assert "→ fix: wait and retry" in rendered


class _Resp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_identity_probe_posts_ping_and_returns_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.backends as backends_module
    from fx1.serve.backends import HostedK3Backend

    captured: dict[str, Any] = {}
    envelope = {
        "id": "cmpl-1",
        "model": "kimi-k3-0905",
        "choices": [{"message": {"role": "assistant", "content": "pong"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def fake_urlopen(request: Any, *, timeout_s: float = 0) -> _Resp:
        captured["body"] = json.loads(request.data.decode())
        captured["headers"] = {name.lower(): value for name, value in request.header_items()}
        captured["timeout"] = timeout_s
        captured["url"] = request.full_url
        return _Resp(json.dumps(envelope).encode())

    monkeypatch.setattr(backends_module, "_openai_urlopen", fake_urlopen)
    backend = HostedK3Backend(api_key="test-key", api_url="https://api.example/v1/chat/completions")
    result = backend.identity_probe()
    assert captured["body"]["max_tokens"] == 1
    assert captured["body"]["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["body"]["temperature"] == 0.0
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["url"] == "https://api.example/v1/chat/completions"
    assert captured["timeout"] == 120.0
    assert result == {
        "model": "kimi-k3-0905",
        "id": "cmpl-1",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    assert backend.last_usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_identity_probe_overrides_sampling_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.backends as backends_module
    from fx1.serve.backends import HostedK3Backend, SamplingParams

    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout_s: float = 0) -> _Resp:
        captured["body"] = json.loads(request.data.decode())
        return _Resp(json.dumps({"id": "i", "model": "m", "usage": None}).encode())

    monkeypatch.setattr(backends_module, "_openai_urlopen", fake_urlopen)
    monkeypatch.setattr(
        SamplingParams,
        "body_fields",
        lambda self: {"temperature": 0.0, "max_tokens": 99},
    )
    backend = HostedK3Backend(api_key="test-key", api_url="https://api.example/v1/chat/completions")
    result = backend.identity_probe(max_tokens=1)
    assert captured["body"]["max_tokens"] == 1
    assert result["model"] == "m"
    assert result["id"] == "i"
    assert result["usage"] is None
    assert backend.last_usage is None


def test_identity_probe_http_401(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.backends as backends_module
    from fx1.serve.backends import HostedK3Backend

    def fake_urlopen(request: Any, *, timeout_s: float = 0) -> _Resp:
        raise urllib.error.HTTPError(
            "https://provider.invalid/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b""),
        )

    monkeypatch.setattr(backends_module, "_openai_urlopen", fake_urlopen)
    backend = HostedK3Backend(api_key="test-key", api_url="https://api.example/v1/chat/completions")
    with pytest.raises(RuntimeError, match=r"^hosted_k3 endpoint returned HTTP 401$"):
        backend.identity_probe()


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (
            urllib.error.URLError(socket.gaierror("dns")),
            r"^hosted_k3 endpoint unreachable \(gaierror\)$",
        ),
        (TimeoutError("slow"), r"^hosted_k3 endpoint transport fault \(TimeoutError\)$"),
    ],
)
def test_identity_probe_transport_errors(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, match: str
) -> None:
    import fx1.serve.backends as backends_module
    from fx1.serve.backends import HostedK3Backend

    def fake_urlopen(request: Any, *, timeout_s: float = 0) -> _Resp:
        raise error

    monkeypatch.setattr(backends_module, "_openai_urlopen", fake_urlopen)
    backend = HostedK3Backend(api_key="test-key", api_url="https://api.example/v1/chat/completions")
    with pytest.raises(RuntimeError, match=match):
        backend.identity_probe()


def test_identity_probe_rejects_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.backends as backends_module
    from fx1.serve.backends import HostedK3Backend

    def fake_urlopen(request: Any, *, timeout_s: float = 0) -> _Resp:
        return _Resp(b"not-json")

    monkeypatch.setattr(backends_module, "_openai_urlopen", fake_urlopen)
    backend = HostedK3Backend(api_key="test-key", api_url="https://api.example/v1/chat/completions")
    with pytest.raises(RuntimeError, match="malformed hosted_k3 completion payload: not JSON"):
        backend.identity_probe()
