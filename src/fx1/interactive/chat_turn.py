"""One conversational model turn: tool rounds, honesty, budget, and cancel.

The concierge keeps the method names the session loop calls and delegates
the turn here. Transcript edits stay on the host so the lock and the
read-only API do not move.
"""

from __future__ import annotations

import threading
from typing import Any

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.chat_tools import CHAT_TOOL_NAMES, chat_tool_schemas, execute_tool_call
from fx1.interactive.model_client import ErrorKind, ModelError
from fx1.interactive.model_ux import (
    CallSnapshot,
    charge_reported_usage,
    completion_kwargs,
    footer_lines,
    open_backend,
    outbound_messages,
    read_snapshot,
    render_model_error,
    scrub_transcript,
)
from fx1.interactive.usage_budget import BudgetExceeded, UsageTracker


def run_chat_turn(host: Any, text: str) -> None:
    """Append the user line and run tool rounds when a backend is open."""
    opened = open_backend(host.model, host._backend_factory)
    if opened.backend is None:
        host._say(opened.detail)
        host._set_status(opened.status)
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
    completion = _call_backend(host, backend, tracker, cancel, tools)
    return _consume_completion(host, completion, tracker, backend)


def _call_backend(
    host: Any,
    backend: Any,
    tracker: UsageTracker,
    cancel: threading.Event,
    tools: list[dict[str, Any]],
) -> Any:
    messages = outbound_messages(host._transcript(), host.model, tools=CHAT_TOOL_NAMES, mode="chat")
    kwargs = completion_kwargs(
        tracker, backend, cancel, pass_deadline=host._pass_deadline, tools=tools
    )
    tracker.charge_step()
    tracker.charge_model_call()
    started = host._clock()
    with host._orb():
        completion = backend.complete_with_tools(messages, **kwargs)
    host._pending_snapshot = read_snapshot(backend, started, host._clock)
    return completion


def _consume_completion(host: Any, completion: Any, tracker: UsageTracker, backend: Any) -> bool:
    content = scrub_transcript(validate_fx1_output(completion.content or ""))
    snapshot = host._pending_snapshot
    if snapshot is not None:
        host._note_call(snapshot)
    if not completion.tool_calls:
        return _finish_answer(host, content, snapshot, tracker, backend)
    charge_reported_usage(tracker, backend)
    _append_tool_round(host, content, completion)
    return False


def _finish_answer(
    host: Any,
    content: str,
    snapshot: CallSnapshot | None,
    tracker: UsageTracker,
    backend: Any,
) -> bool:
    host._append_message({"role": "assistant", "content": content})
    host._say(content)
    _say_footer(host, snapshot)
    host._remember_answer(content)
    charge_reported_usage(tracker, backend)
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


def _apply_tool_calls(host: Any, completion: Any) -> None:
    env = host._tool_env(host._current_cancel())
    for call in completion.tool_calls:
        result_text = scrub_transcript(execute_tool_call(call, env))
        host._append_message({"role": "tool", "tool_call_id": call["id"], "content": result_text})


def _withhold(host: Any, start: int, exc: Fx1HonestyError) -> None:
    refusal = f"answer withheld by the fx-1 honesty gate ({exc}). Ask a follow-up to continue."
    if any(message.get("role") == "tool" for message in host._messages[start:]):
        # A blocked answer cannot undo completed tool actions. Keep
        # their validated call/result transcript for the next turn.
        host._append_message({"role": "assistant", "content": refusal})
    else:
        host._truncate_messages(start)
    host._say(refusal)
    host._set_status("withheld")


def _fail_model(host: Any, start: int, exc: ModelError) -> None:
    if exc.kind is ErrorKind.CANCELLED:
        _cancel_marked(host, start)
        return
    host._truncate_messages(start)
    if exc.kind is ErrorKind.DEADLINE:
        host._say(str(exc))
        host._set_status("budget")
        return
    host._say_error(RuntimeError(render_model_error(exc)))
    host._set_status("error")
    host._emit("concierge.model_error", {"kind": str(exc.kind), "status": exc.status})


def _fail_generic(host: Any, start: int, exc: Exception) -> None:
    host._truncate_messages(start)
    host._say_error(exc)
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


def _say_footer(host: Any, snapshot: CallSnapshot | None) -> None:
    latency = None if snapshot is None else snapshot.latency_s
    usage = None if snapshot is None else snapshot.usage
    identity = None if snapshot is None else snapshot.identity
    for line in footer_lines(host.model, latency, usage, identity):
        host._say(line)


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
            copied["function"] = fn
        scrubbed.append(copied)
    return scrubbed
