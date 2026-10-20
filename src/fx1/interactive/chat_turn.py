"""One conversational model turn: tool rounds, honesty, budget, and cancel.

The concierge keeps the method names the session loop calls and delegates
the turn here. Transcript edits stay on the host so the lock and the
read-only API do not move.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from fx1.honesty import (
    MODEL_TEXT_WITHHELD,
    Fx1HonestyError,
    screen_piece,
    scrub_model_text,
    validate_fx1_output,
)
from fx1.interactive.chat_tools import CHAT_TOOL_NAMES, chat_tool_schemas, execute_tool_call
from fx1.interactive.evidence import EvidenceError
from fx1.interactive.evidence import ground_summary as ground_summary
from fx1.interactive.model_client import ErrorKind, ModelError
from fx1.interactive.model_ux import (
    CallSnapshot,
    Outbound,
    build_outbound,
    charge_reported_usage,
    completion_kwargs,
    footer_lines,
    open_backend,
    read_snapshot,
    render_model_error,
    scrub_transcript,
)
from fx1.interactive.usage_budget import BudgetExceeded, UsageTracker


class _CancelledTurn(Exception):
    """The operator cancelled between steps. Not a model failure."""


def run_chat_turn(host: Any, text: str) -> None:
    """Append the user line and run tool rounds when a backend is open."""
    opened = open_backend(host.model, host._backend_factory)
    if opened.backend is None:
        host._say(opened.detail)
        host._set_status(opened.status)
        return
    if _cancelled_now(host):
        _cancel_marked(host, len(host._messages))
        return
    host._turn_mark = len(host._messages)
    host._append_message({"role": "user", "content": scrub_transcript(text)})
    _run_turn(host, opened.backend)


def cancel_chat_keyboard(host: Any) -> None:
    """Ctrl-C outside a tool round: drop the partial turn and stay in the session."""
    start = host._turn_mark
    host.cancel_turn()
    if start is not None:
        host._truncate_messages(start)
        host._turn_mark = None
    host._say("turn cancelled")
    host._set_status("cancelled")


def _run_turn(host: Any, backend: Any) -> None:
    start = host._turn_mark if host._turn_mark is not None else len(host._messages)
    try:
        _tool_rounds(host, backend)
    except KeyboardInterrupt:
        _cancel_marked(host, start)
    except _CancelledTurn:
        _cancel_marked(host, start)
    except Fx1HonestyError as exc:
        _withhold(host, start, exc)
    except BudgetExceeded:
        _stop_for_budget(host)
    except ModelError as exc:
        _fail_model(host, start, exc)
    except Exception as exc:  # noqa: BLE001 — a dead network must not kill the shell
        _fail_generic(host, start, exc)
    finally:
        host._turn_mark = None
        host._active_tracker = None


def _tool_rounds(host: Any, backend: Any) -> None:
    tracker = UsageTracker(host._usage_preset, monotonic=host._clock)
    tracker.start()
    host._active_tracker = tracker
    cancel = threading.Event()
    host._arm_turn(cancel)
    try:
        _round_loop(host, backend, tracker, cancel)
    except KeyboardInterrupt:
        cancel.set()
        raise
    finally:
        host._disarm_turn()


def _round_loop(host: Any, backend: Any, tracker: UsageTracker, cancel: threading.Event) -> None:
    tools = chat_tool_schemas()
    for _ in range(12):
        if _one_round(host, backend, tracker, cancel, tools):
            return
    host._say("(tool loop budget exhausted — ask a follow-up to continue)")
    host._set_status("budget")


def _one_round(
    host: Any,
    backend: Any,
    tracker: UsageTracker,
    cancel: threading.Event,
    tools: list[dict[str, Any]],
) -> bool:
    _raise_if_cancelled(host)
    completion = _call_backend(host, backend, tracker, cancel, tools)
    return _consume_completion(host, completion, tracker, backend)


def _call_backend(
    host: Any,
    backend: Any,
    tracker: UsageTracker,
    cancel: threading.Event,
    tools: list[dict[str, Any]],
) -> Any:
    kwargs = completion_kwargs(
        tracker, backend, cancel, pass_deadline=host._pass_deadline, tools=tools
    )
    tracker.charge_step()
    retries = 0
    while True:
        note = _dataset_note(host)
        outbound = build_outbound(
            _fold_dataset_note(host._transcript(), note),
            host.model,
            tools=CHAT_TOOL_NAMES,
            mode="chat",
            max_chars=host._context_budget.limit(host.model),
        )
        _announce_trim(host, outbound)
        tracker.charge_model_call()
        started = host._clock()
        try:
            with host._orb():
                completion = backend.complete_with_tools(outbound.messages, **kwargs)
        except ModelError as exc:
            retries += 1
            if retries > _CONTEXT_RETRIES or not _shrunk_for_retry(host, exc, outbound):
                raise
            continue
        host._pending_snapshot = read_snapshot(backend, started, host._clock)
        return completion


_CONTEXT_RETRIES = 3


def _dataset_note(host: Any) -> str:
    provider = getattr(host, "dataset_model_note", None)
    if not callable(provider):
        return ""
    note = provider()
    return note if isinstance(note, str) else ""


def _fold_dataset_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Put the dataset note in the latest user turn so it counts toward the fit."""
    if not note:
        return list(messages)
    copied = [dict(item) for item in messages]
    index = _last_user(copied)
    if index is None:
        return [*copied, {"role": "user", "content": note}]
    message = copied[index]
    content = message.get("content")
    body = content if isinstance(content, str) else ""
    copied[index] = {**message, "content": f"{note}\n\n{body}"}
    return copied


def _last_user(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _announce_trim(host: Any, outbound: Outbound) -> None:
    if not outbound.omitted:
        host._context_trim_said = False
        return
    if host._context_trim_said:
        return
    host._context_trim_said = True
    host._say(
        f"context: {outbound.omitted} older messages are no longer sent to {host.model} "
        "so the conversation fits; they stay in the saved session"
    )


def _shrunk_for_retry(host: Any, exc: ModelError, outbound: Outbound) -> bool:
    """True when a too-long prompt can be retried with fewer earlier turns."""
    if exc.kind is not ErrorKind.CONTEXT_TOO_LONG or outbound.turns < 2:
        return False
    if not host._context_budget.shrink(host.model, outbound.chars):
        return False
    host._say(f"context: {host.model} rejected the prompt as too long; retrying with less history")
    return True


_MAX_TOOL_CALLS = 32


def _consume_completion(host: Any, completion: Any, tracker: UsageTracker, backend: Any) -> bool:
    if _refuse_tool_burst(host, completion):
        return True
    content = scrub_transcript(completion.content or "")
    _require_clean(host, content)
    _require_clean_calls(host, completion)
    snapshot = host._pending_snapshot
    if snapshot is not None:
        host._note_call(snapshot)
    if not completion.tool_calls:
        return _finish_answer(host, content, snapshot, tracker, backend)
    charge_reported_usage(tracker, backend)
    _append_tool_round(host, content, completion)
    return False


def _refuse_tool_burst(host: Any, completion: Any) -> bool:
    calls = getattr(completion, "tool_calls", None)
    if not calls or len(calls) <= _MAX_TOOL_CALLS:
        return False
    host._say("too many tool calls in one response")
    host._set_status("error")
    return True


def _finish_answer(
    host: Any,
    content: str,
    snapshot: CallSnapshot | None,
    tracker: UsageTracker,
    backend: Any,
) -> bool:
    shown = _shown_answer(content, _verified_refs(host))
    _publish_answer(host, shown, content, snapshot, tracker, backend)
    return True


def _append_tool_round(host: Any, content: str, completion: Any) -> None:
    host._append_message(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": _scrub_tool_calls(completion.tool_calls),
        }
    )
    _apply_tool_calls(host, completion)


class _GateBlock:
    def __init__(self) -> None:
        self.flag = False


def _visible_contents(host: Any) -> list[str]:
    """Assistant lines only, so a tool envelope cannot separate a split claim."""
    parts: list[str] = []
    for message in host._messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return parts


def _require_clean(host: Any, text: str) -> None:
    validate_fx1_output(text)
    if screen_piece(_visible_contents(host), text) != text:
        raise Fx1HonestyError("fx-1 output failed the honesty gate")


def _require_clean_calls(host: Any, completion: Any) -> None:
    for call in completion.tool_calls or ():
        _require_call(host, call)


def _require_call(host: Any, call: Any) -> None:
    if not isinstance(call, dict):
        return
    function = call.get("function")
    if not isinstance(function, dict):
        return
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        _require_clean(host, arguments)


def _apply_tool_calls(host: Any, completion: Any) -> None:
    env = host._tool_env(host._current_cancel())
    block = _GateBlock()
    _guard_says(host, env, block)
    for call in completion.tool_calls:
        _raise_if_cancelled(host)
        block.flag = False
        result_text = _tool_result(host, call, env, block)
        host._append_message({"role": "tool", "tool_call_id": call["id"], "content": result_text})


def _tool_result(host: Any, call: Any, env: Any, block: _GateBlock) -> str:
    result_text = scrub_transcript(execute_tool_call(call, env))
    if block.flag or screen_piece(_visible_contents(host), result_text) != result_text:
        return MODEL_TEXT_WITHHELD
    return result_text


def _guard_says(host: Any, env: Any, block: _GateBlock) -> None:
    say = env.say
    say_error = env.say_error

    def emit(text: str) -> None:
        say(_shown(host, text, block))

    def emit_error(exc: Exception) -> None:
        cleaned = _shown(host, str(exc), block)
        if cleaned != str(exc):
            say_error(RuntimeError(cleaned))
            return
        say_error(exc)

    env.say = emit
    env.say_error = emit_error


def _shown(host: Any, text: str, block: _GateBlock) -> str:
    cleaned = screen_piece(_visible_contents(host), text)
    if cleaned != text:
        block.flag = True
    return cleaned


def _cancelled_now(host: Any) -> bool:
    checker = getattr(host, "_turn_cancelled", None)
    return bool(callable(checker) and checker())


def _raise_if_cancelled(host: Any) -> None:
    if _cancelled_now(host):
        raise _CancelledTurn()


def _refusal_text(exc: Fx1HonestyError) -> str:
    """Operator text for a withhold. The sentence itself must pass the gate."""
    detail = str(exc)
    if scrub_model_text(detail) != detail:
        return MODEL_TEXT_WITHHELD
    return f"answer withheld by the fx-1 honesty gate ({detail}). Ask a follow-up to continue."


def _withhold(host: Any, start: int, exc: Fx1HonestyError) -> None:
    refusal = _refusal_text(exc)
    if any(message.get("role") == "tool" for message in host._messages[start:]):
        # A blocked answer cannot undo completed tool actions. Keep
        # their validated call/result transcript for the next turn.
        host._append_message({"role": "assistant", "content": refusal})
    else:
        host._truncate_messages(start)
    host._say(refusal)
    host._set_status("withheld")
    remember = getattr(host, "_remember_withheld", None)
    if callable(remember):
        remember(refusal)


def _fail_model(host: Any, start: int, exc: ModelError) -> None:
    if exc.kind is ErrorKind.CANCELLED:
        _cancel_marked(host, start)
        return
    host._truncate_messages(start)
    if exc.kind is ErrorKind.DEADLINE:
        host._say(scrub_model_text(str(exc)))
        host._set_status("budget")
        return
    host._say_error(RuntimeError(render_model_error(exc)))
    host._set_status("error")
    host._emit("concierge.model_error", {"kind": str(exc.kind), "status": exc.status})


def _fail_generic(host: Any, start: int, exc: Exception) -> None:
    host._truncate_messages(start)
    host._say_error(RuntimeError(scrub_model_text(str(exc))))
    host._set_status("error")


def _stop_for_budget(host: Any) -> None:
    tracker = host._active_tracker
    detail = tracker.describe() if tracker is not None else "usage ceiling reached"
    host._say(f"turn stopped by usage ceiling. {detail}")
    host._set_status("budget")


def _cancel_marked(host: Any, start: int) -> None:
    host.cancel_turn()
    host._truncate_messages(start)
    host._say("turn cancelled")
    host._set_status("cancelled")


_METRIC = re.compile(
    r"\b(?:sharpe|sortino|calmar|nav|pnl)\b|\bp\s*&\s*l\b",
    re.IGNORECASE,
)


def _shown_answer(content: str, verified: set[str]) -> str:
    grounded = _ground_visible(content, verified)
    try:
        return validate_fx1_output(grounded)
    except Fx1HonestyError:
        return MODEL_TEXT_WITHHELD


def _ground_visible(text: str, verified: set[str]) -> str:
    """Rewrite a forged verification claim. Bare metric discussion stays."""
    try:
        return ground_summary(text, verified_receipts=verified)
    except EvidenceError:
        return _ground_after_metric(text, verified)


def _ground_after_metric(text: str, verified: set[str]) -> str:
    masked = _METRIC.sub("metric", text)
    if masked == text:
        return MODEL_TEXT_WITHHELD
    try:
        rewritten = ground_summary(masked, verified_receipts=verified)
    except EvidenceError:
        return MODEL_TEXT_WITHHELD
    if rewritten == masked:
        return text
    return MODEL_TEXT_WITHHELD


def _verified_refs(host: Any) -> set[str]:
    ledger = getattr(host, "_evidence", None)
    take = getattr(ledger, "verified_receipts", None)
    if not callable(take):
        return set()
    return _ref_set(take())


def _ref_set(items: Any) -> set[str]:
    found: set[str] = set()
    for item in items:
        ref = getattr(item, "ref", "")
        if isinstance(ref, str) and ref:
            found.add(ref)
    return found


def _publish_answer(
    host: Any,
    shown: str,
    content: str,
    snapshot: CallSnapshot | None,
    tracker: UsageTracker,
    backend: Any,
) -> None:
    host._append_message({"role": "assistant", "content": shown})
    host._say(shown)
    if shown == MODEL_TEXT_WITHHELD and shown != content:
        _note_withheld(host, shown)
        charge_reported_usage(tracker, backend)
        return
    _say_footer(host, snapshot)
    host._remember_answer(shown)
    charge_reported_usage(tracker, backend)


def _note_withheld(host: Any, shown: str) -> None:
    host._set_status("withheld")
    remember = getattr(host, "_remember_withheld", None)
    if callable(remember):
        remember(shown)


def _say_footer(host: Any, snapshot: CallSnapshot | None) -> None:
    latency = None if snapshot is None else snapshot.latency_s
    usage = None if snapshot is None else snapshot.usage
    identity = None if snapshot is None else snapshot.identity
    for line in footer_lines(host.model, latency, usage, identity):
        host._say(scrub_model_text(line))


def _scrub_tool_calls(calls: Any) -> list[dict[str, Any]]:
    """Copy tool calls with credential values removed from argument JSON."""
    scrubbed: list[dict[str, Any]] = []
    for call in calls or ():
        if not isinstance(call, dict):
            continue
        copied = dict(call)
        function = copied.get("function")
        if isinstance(function, dict):
            fn = dict(function)
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                fn["arguments"] = scrub_transcript(arguments)
                if screen_piece([], fn["arguments"]) != fn["arguments"]:
                    fn["arguments"] = "{}"
            copied["function"] = fn
        scrubbed.append(copied)
    return scrubbed
