"""Provider usage validation at the real HTTP-decoding boundary."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from fx1.serve import backends


@pytest.mark.parametrize(
    "value",
    [True, False, "3", 3.0, 3.5, float("nan"), float("inf"), float("-inf"), None],
)
def test_provider_usage_accepts_only_integer_counts(value: Any) -> None:
    """Malformed counters are dropped before they can be charged or accumulated."""
    payload = {"usage": {"prompt_tokens": 7, "completion_tokens": value, "cached_tokens": -2}}
    assert backends._extract_usage(payload) == {"prompt_tokens": 7, "cached_tokens": -2}


@pytest.mark.parametrize("usage", [None, {}, [], "bad", {"total_tokens": True}])
def test_absent_or_unbillable_usage_stays_absent(usage: Any) -> None:
    assert backends._extract_usage({"usage": usage}) is None


def test_genuine_integer_usage_is_not_coerced_or_dropped() -> None:
    usage = {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2, "cached_tokens": -1}
    assert backends._extract_usage({"usage": usage}) == usage


@pytest.mark.parametrize("value", [3.5, float("nan"), float("inf"), float("-inf")])
def test_malformed_usage_does_not_discard_a_completed_response(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    """Exercise the provider parser, not an injected backend that skips it."""
    payload = {
        "choices": [{"message": {"content": "completed"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": value},
    }

    def response(*args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(backends, "_openai_urlopen", response)
    content, usage = backends._openai_chat_complete(
        "https://provider.example/v1/chat/completions",
        model="synthetic-model",
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=1.0,
        api_key=None,
        label="SYNTHETIC",
    )
    assert content == "completed"
    assert usage == {"prompt_tokens": 5}


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": 3.0, "completion_tokens": 4.0},
        {"total_tokens": True},
        {"total_tokens": float("nan")},
        {"total_tokens": float("inf")},
        {"note": "unbillable"},
    ],
)
def test_usage_only_stream_frame_does_not_require_billable_counters(
    monkeypatch: pytest.MonkeyPatch, usage: dict[str, Any]
) -> None:
    frames = [
        {"choices": [{"delta": {"content": "completed"}}]},
        {"usage": usage},
    ]
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"

    def response(*args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(body.encode())

    monkeypatch.setattr(backends, "_openai_urlopen", response)
    usage_out: list[dict[str, int]] = []
    chunks = list(
        backends._openai_chat_stream(
            "https://provider.example/v1/chat/completions",
            model="synthetic-model",
            messages=[{"role": "user", "content": "hello"}],
            timeout_s=1.0,
            api_key=None,
            label="SYNTHETIC",
            usage_out=usage_out,
        )
    )
    assert chunks == ["completed"]
    assert usage_out == []


@pytest.mark.parametrize("frame", [{}, {"usage": None}, {"usage": "bad"}, {"usage": []}])
def test_non_usage_stream_frame_without_choices_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, frame: dict[str, Any]
) -> None:
    def response(*args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n".encode())

    monkeypatch.setattr(backends, "_openai_urlopen", response)
    with pytest.raises(RuntimeError, match="missing choices"):
        list(
            backends._openai_chat_stream(
                "https://provider.example/v1/chat/completions",
                model="synthetic-model",
                messages=[{"role": "user", "content": "hello"}],
                timeout_s=1.0,
                api_key=None,
                label="SYNTHETIC",
            )
        )
