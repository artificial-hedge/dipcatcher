"""SYNTHETIC independent checks of the cited-completion gate (chat.py).

Deterministic in-process backends: the honesty gate's position in the
pipeline (truncate → validate → footer) and the fail-closed tool channel.
"""

from __future__ import annotations

from typing import Any

import pytest

from fx1.honesty import Fx1HonestyError
from fx1.serve.backends import SamplingParams, ToolCompletion
from fx1.serve.chat import cited_complete, cited_complete_tools

_SHA = "f" * 64


class _Plain:
    """A backend with only the plain channel."""

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        return self._text


class _Toolful(_Plain):
    """A backend with the structured channel."""

    def __init__(self, text: str | None, **result_kw: Any) -> None:
        super().__init__(text or "")
        self._kw = result_kw

    def complete_with_tools(self, *args: Any, **kwargs: Any) -> ToolCompletion:
        return ToolCompletion(**{"content": self._text, "finish_reason": "stop", **self._kw})


def test_cited_complete_gates_forbidden_headlines() -> None:
    for claim in ("the sharpe is 2.4", "pnl: +$4,200", "navs hit 1.9"):
        with pytest.raises(Fx1HonestyError):
            cited_complete(_Plain(claim), [])


def test_stop_truncates_before_the_gate() -> None:
    # the shipped bytes are what get validated — a stop cutting the
    # forbidden claim before it ships is the contract working as designed
    out = cited_complete(
        _Plain("clean part. sharpe 9.9 trailing"),
        [],
        sampling=SamplingParams(stop=("sharpe",)),
    )
    assert out == "clean part. "
    with pytest.raises(Fx1HonestyError):
        cited_complete(_Plain("the sharpe is 9.9"), [])  # no stop → gate sees it
    # bare discussion without a numeric claim is discussion, not a headline
    assert cited_complete(_Plain("why sharpe is forbidden"), []) == "why sharpe is forbidden"


def test_receipt_footer_appends_after_validation() -> None:
    out = cited_complete(_Plain("result"), [], receipt_hashes=[_SHA, "b" * 64])
    assert (
        out
        == f"result\n\nEvidence: `{'f' * 16}…`, `{'b' * 16}…` — verify with `dipcatcher verify-research`."
    )
    assert cited_complete(_Plain("result"), [], receipt_hashes=[]) == "result"
    assert cited_complete(_Plain("result"), []) == "result"


def test_honesty_error_propagates_not_wrapped() -> None:
    class _Refusing(_Plain):
        def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
            raise Fx1HonestyError("nope")

    with pytest.raises(Fx1HonestyError):
        cited_complete(_Refusing(""), [])


def test_tools_channel_fails_closed_notimplemented() -> None:
    with pytest.raises(NotImplementedError):
        cited_complete_tools(
            _Plain("x"), [], tools=[{"type": "function", "function": {"name": "f"}}]
        )


def test_tools_channel_full_contract() -> None:
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    backend = _Toolful(
        "text END ignored", tool_calls=calls, finish_reason="tool_calls", logprobs={"content": []}
    )
    out = cited_complete_tools(
        backend,
        [],
        receipt_hashes=[_SHA],
        sampling=SamplingParams(stop=(" END",)),
    )
    # truncation applies to content; tool_calls/logprobs ride verbatim
    assert (
        out.content
        == "text\n\nEvidence: `ffffffffffffffff…` — verify with `dipcatcher verify-research`."
    )
    assert out.tool_calls is calls
    assert out.finish_reason == "tool_calls"
    assert out.logprobs == {"content": []}


def test_tools_channel_footers_a_none_content() -> None:
    out = cited_complete_tools(
        _Toolful(None, tool_calls=[], finish_reason="stop"),
        [],
        receipt_hashes=[_SHA],
    )
    assert out.content is not None and "Evidence:" in out.content


def test_tools_channel_gates_text_but_not_machine_json() -> None:
    # a sharpe key inside tool arguments is machine-bound JSON — the gate
    # reads content only; the claim surface stays gated regardless
    calls = [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"sharpe": 2}'}}
    ]
    out = cited_complete_tools(_Toolful(None, tool_calls=calls), [], tools=[])
    assert out.tool_calls is calls
    with pytest.raises(Fx1HonestyError):
        cited_complete_tools(_Toolful("my sortino is 4", tool_calls=calls), [])


def test_unchanged_content_returns_the_same_object() -> None:
    result = _Toolful("plain", tool_calls=None, finish_reason="stop", logprobs=None)
    out = cited_complete_tools(result, [])
    # no truncation, no footer → the ToolCompletion passes through untouched
    assert out.content == "plain" and out.tool_calls is None
