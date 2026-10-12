"""Long conversations: outbound fitting, the omission note, and the too-long retry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import profiles
from fx1.interactive.concierge import Concierge
from fx1.interactive.context_window import (
    DEFAULT_MAX_CHARS,
    ENV_MAX_CHARS,
    MIN_MAX_CHARS,
    ContextBudget,
    fit_dialogue,
    message_chars,
    omission_note,
)
from fx1.interactive.model_client import ModelClient, RetryPolicy
from fx1.interactive.model_ux import build_outbound, outbound_messages
from tests.fx1.stub_model_server import Reply, StubModelServer

KEY = "ctx-test-key-aaaa"
_TINY_RETRY = RetryPolicy(
    max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0, honor_retry_after_up_to_s=1.0
)


def _turn(index: int, size: int = 100) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"q{index} " + "x" * size},
        {"role": "assistant", "content": f"a{index} " + "y" * size},
    ]


def _dialogue(turns: int, size: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index in range(turns):
        out.extend(_turn(index, size))
    return out


def _tool_turn() -> list[dict[str, Any]]:
    call = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
    return [
        {"role": "user", "content": "use the tool " + "z" * 200},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "result " + "r" * 200},
        {"role": "assistant", "content": "done"},
    ]


def test_a_dialogue_under_the_ceiling_is_sent_whole() -> None:
    dialogue = _dialogue(3)
    fitted = fit_dialogue(dialogue, 10_000)
    assert fitted.messages == dialogue
    assert fitted.omitted == 0
    assert fitted.turns == 3
    assert fitted.chars == sum(message_chars(m) for m in dialogue)


def test_the_oldest_whole_turns_go_first_and_the_newest_stays() -> None:
    dialogue = _dialogue(6)
    one_turn = sum(message_chars(m) for m in _turn(0))
    fitted = fit_dialogue(dialogue, one_turn * 2 + 5)
    assert fitted.omitted == 8
    assert fitted.turns == 2
    assert fitted.messages == dialogue[8:]
    assert fitted.messages[0]["role"] == "user"
    assert fitted.chars <= one_turn * 2 + 5


def test_the_newest_turn_is_kept_even_when_it_alone_is_too_big() -> None:
    dialogue = _dialogue(3) + _turn(9, size=5000)
    fitted = fit_dialogue(dialogue, 500)
    assert fitted.turns == 1
    assert fitted.messages == dialogue[-2:]
    assert fitted.omitted == len(dialogue) - 2


def test_a_tool_call_is_never_separated_from_its_result() -> None:
    dialogue = _dialogue(2) + _tool_turn() + _dialogue(1)
    for ceiling in range(1, 2000, 37):
        fitted = fit_dialogue(dialogue, ceiling)
        roles = [m["role"] for m in fitted.messages]
        assert roles[0] == "user"
        for index, message in enumerate(fitted.messages):
            if message["role"] == "tool":
                assert fitted.messages[index - 1].get("tool_calls") or (
                    fitted.messages[index - 1]["role"] == "tool"
                )
        assert fitted.messages == dialogue[fitted.omitted :]


def test_a_dialogue_that_starts_with_an_assistant_message_still_fits() -> None:
    dialogue = [{"role": "assistant", "content": "welcome " * 50}, *_dialogue(2)]
    fitted = fit_dialogue(dialogue, message_chars(dialogue[-1]) * 3)
    assert fitted.messages[0]["role"] == "user"
    assert fitted.omitted >= 1


def test_an_empty_dialogue_is_empty() -> None:
    fitted = fit_dialogue([], 10)
    assert fitted.messages == []
    assert fitted.omitted == 0
    assert fitted.turns == 1


def test_fitting_does_not_edit_the_stored_transcript() -> None:
    dialogue = _dialogue(5)
    before = [dict(m) for m in dialogue]
    fit_dialogue(dialogue, 300)
    assert dialogue == before


def test_budget_defaults_and_env_parsing() -> None:
    assert ContextBudget().limit("fx1") == DEFAULT_MAX_CHARS
    assert ContextBudget.from_env({}).limit("fx1") == DEFAULT_MAX_CHARS
    assert ContextBudget.from_env({ENV_MAX_CHARS: "50000"}).limit("fx1") == 50_000
    assert ContextBudget.from_env({ENV_MAX_CHARS: "12"}).limit("fx1") == MIN_MAX_CHARS
    for bad in ("", "abc", "-5", "1e6", "٣٠٠٠٠", "5 000", "0x10"):
        assert ContextBudget.from_env({ENV_MAX_CHARS: bad}).limit("fx1") == DEFAULT_MAX_CHARS


def test_shrink_halves_what_was_sent_per_model_down_to_a_floor() -> None:
    budget = ContextBudget(100_000)
    assert budget.shrink("fx1", 40_000) is True
    assert budget.limit("fx1") == 20_000
    assert budget.limit("fx1-lite") == 100_000
    assert budget.shrink("fx1", 20_000) is True
    assert budget.limit("fx1") == 10_000
    seen = 0
    while budget.shrink("fx1", budget.limit("fx1")):
        seen += 1
        assert seen < 20
    assert budget.limit("fx1") == MIN_MAX_CHARS
    assert budget.shrink("fx1", 1) is False


def test_build_outbound_without_a_ceiling_matches_outbound_messages() -> None:
    transcript = _dialogue(4)
    built = build_outbound(transcript, "fx1", tools=(), mode="chat")
    assert built.messages == outbound_messages(transcript, "fx1", tools=(), mode="chat")
    assert built.omitted == 0


def test_build_outbound_tells_the_model_what_was_left_out() -> None:
    transcript = _dialogue(8)
    built = build_outbound(transcript, "fx1", mode="chat", max_chars=600)
    assert built.omitted > 0
    assert built.messages[0]["role"] == "system"
    assert omission_note(built.omitted) in built.messages[0]["content"]
    assert built.messages[1]["role"] == "user"
    assert built.messages[1:] == transcript[built.omitted :]
    full = build_outbound(transcript, "fx1", mode="chat")
    assert omission_note(1)[:20] not in full.messages[0]["content"]


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.delenv(ENV_MAX_CHARS, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def _factory(**kwargs: Any) -> ModelClient:
    return ModelClient(retry=_TINY_RETRY, sleep=lambda _d: None, timeout_s=2.0, **kwargs)


def _concierge(out: list[str], srv: StubModelServer, ceiling: int) -> Concierge:
    profiles.set_endpoint("fx1", KEY, srv.url)
    concierge = Concierge(
        backend_factory=_factory,
        output_fn=out.append,
        animated=False,
        input_fn=lambda _p="": "",
    )
    concierge._context_budget = ContextBudget(ceiling)
    return concierge


def test_a_long_chat_sends_a_window_and_says_so_once(env: Path) -> None:
    del env
    out: list[str] = []
    with StubModelServer({"fx1": KEY}) as srv:
        concierge = _concierge(out, srv, ceiling=MIN_MAX_CHARS)
        for index in range(12):
            concierge.handle_line(f"turn {index} " + "p" * 900)
        requests = srv.requests
    assert len(requests) == 12
    sizes = [r.n_messages for r in requests]
    assert max(sizes) < 1 + 2 * 11 + 1
    assert sizes[-1] <= sizes[-2] + 2
    text = "\n".join(out)
    assert text.count("older messages are no longer sent to fx1") == 1
    assert len(concierge.messages()) == 24
    assert concierge.last_status == "ok"
    last = requests[-1].body
    assert last is not None
    assert "left out of this request" in last["messages"][0]["content"]
    assert last["messages"][-1]["role"] == "user"
    assert "turn 11" in last["messages"][-1]["content"]


def test_a_context_too_long_reply_retries_with_less_history(env: Path) -> None:
    del env
    out: list[str] = []
    too_long = Reply.error(
        400,
        body={
            "error": {
                "message": "This model's maximum context length is exceeded.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
    )
    with StubModelServer({"fx1": KEY}) as srv:
        concierge = _concierge(out, srv, ceiling=DEFAULT_MAX_CHARS)
        for index in range(6):
            concierge.handle_line(f"warm {index} " + "w" * 800)
        srv.queue(too_long)
        concierge.handle_line("the question " + "q" * 800)
        requests = srv.requests
    assert concierge.last_status == "ok"
    assert concierge.last_answer == "stub fx1 reply"
    rejected, retried = requests[-2], requests[-1]
    assert retried.n_messages < rejected.n_messages
    text = "\n".join(out)
    assert "rejected the prompt as too long; retrying with less history" in text
    assert concierge._context_budget.limit("fx1") < DEFAULT_MAX_CHARS
    assert len(concierge.messages()) == 14
    assert concierge.messages()[-2]["content"].startswith("the question")


def test_a_context_error_on_a_single_turn_is_reported_not_retried(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del env
    out: list[str] = []
    too_long = Reply.error(
        400, body={"error": {"message": "context length exceeded", "type": "x", "code": "x"}}
    )
    with StubModelServer({"fx1": KEY}) as srv:
        concierge = _concierge(out, srv, ceiling=DEFAULT_MAX_CHARS)
        srv.queue(too_long)
        concierge.handle_line("only turn " + "o" * 500)
        requests = srv.requests
    assert len(requests) == 1
    assert concierge.last_status == "error"
    assert "exceeds the model context" in capsys.readouterr().err
    assert concierge.messages() == []
