"""Terminal-escape injection through concierge, console, and writer output.

Model answers, tool text, flash entries, job events, and errors are hostile.
Display strips ESC/CSI/OSC, C1, bidi overrides, and line-overwrite controls
after the honesty gate has seen the original text.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from fx1.flash.store import FlashStore
from fx1.interactive import concierge as concierge_module
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import BackgroundJob, Concierge
from fx1.interactive.console import _show_error
from fx1.interactive.superpower import CapabilityRun, SuperpowerPlan, SuperpowerResult
from fx1.interactive.terminal import ConsoleWriter, Style, TermCaps, sanitize_for_terminal
from fx1.serve.backends import ToolCompletion
from tests.fx1.test_concierge import FakeBackend, FakeHarness, FakeHarnessResult

_OSC_TITLE = "\x1b]0;owned\x07"
_OSC_LINK = "\x1b]8;;https://evil.example/x\x07click\x1b]8;;\x07"
_CSI_CLEAR = "\x1b[2J\x1b[H"
_C1 = "\x9b31m"
_BIDI = "\u202eABC\u202c"
_ATTACK = f"pinball {_OSC_TITLE}{_OSC_LINK}{_CSI_CLEAR}{_C1}{_BIDI}\rEVIL\x08Z\u2028NL\u061c"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def _caps(*, color: bool) -> TermCaps:
    return TermCaps(
        stdin_tty=True,
        stdout_tty=True,
        interactive=True,
        color=color,
        unicode=True,
        width=80,
        height=24,
        windows=False,
    )


def _controls_gone(text: str) -> None:
    assert "\x1b" not in text
    assert "\x9b" not in text
    assert "\u202e" not in text
    assert "\u2028" not in text
    assert "\u2029" not in text
    assert "\u061c" not in text
    assert "\r" not in text
    assert "\x08" not in text
    assert "owned" not in text
    assert "evil.example" not in text


def _concierge(out: list[str], **kwargs: Any) -> Concierge:
    return Concierge(
        output_fn=out.append,
        animated=False,
        input_fn=lambda _prompt: "",
        session_store=None,
        approvals=kwargs.pop("approvals", ApprovalGate()),
        **kwargs,
    )


def test_model_answer_is_sanitized_after_the_honesty_gate(tmp_path: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")
    text = f"pinball loss stayed finite {_ATTACK}"
    backend = FakeBackend([ToolCompletion(content=text, tool_calls=None, finish_reason="stop")])
    out: list[str] = []
    concierge = _concierge(out, backend_factory=lambda **_ignored: backend)
    concierge.handle_line("score it")
    shown = "\n".join(out)
    _controls_gone(shown)
    assert "pinball loss stayed finite" in shown
    assert "\x1b]0;owned" in concierge.last_answer
    assert concierge.last_status == "ok"


def test_honesty_gate_still_sees_a_metric_hidden_beside_escapes(tmp_path: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")
    text = f"The Sharpe ratio is 2. {_OSC_TITLE}"
    backend = FakeBackend([ToolCompletion(content=text, tool_calls=None, finish_reason="stop")])
    out: list[str] = []
    concierge = _concierge(out, backend_factory=lambda **_ignored: backend)
    concierge.handle_line("score it")
    assert concierge.last_status == "withheld"
    assert "owned" not in "\n".join(out)
    assert "\x1b" not in "\n".join(out)


def test_say_error_strips_controls_on_stderr() -> None:
    out: list[str] = []
    concierge = _concierge(out)
    err = io.StringIO()
    previous = sys.stderr
    sys.stderr = err
    try:
        concierge._say_error(RuntimeError(_ATTACK))  # noqa: SLF001
    finally:
        sys.stderr = previous
    _controls_gone(err.getvalue())
    assert "pinball" in err.getvalue()


def test_harness_stdout_and_stderr_cannot_inject_controls(tmp_path: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")

    class Hostile(FakeHarness):
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            self.calls.append((name, extra_args))
            return FakeHarnessResult(stdout=f"pinball {_ATTACK}", stderr=f"boom {_ATTACK}")

    harness = Hostile()
    call = {
        "id": "h1",
        "type": "function",
        "function": {
            "name": "harness_run",
            "arguments": json.dumps({"name": "doctor", "args": []}),
        },
    }
    backend = FakeBackend(
        [
            ToolCompletion(content=None, tool_calls=(call,), finish_reason="tool_calls"),
            ToolCompletion(content="noted", tool_calls=None, finish_reason="stop"),
        ]
    )
    out: list[str] = []
    err = io.StringIO()
    previous = sys.stderr
    sys.stderr = err
    try:
        concierge = _concierge(
            out,
            harness=harness,
            backend_factory=lambda **_ignored: backend,
            store=FlashStore(tmp_path / "flash.jsonl"),
        )
        concierge.handle_line("run doctor")
    finally:
        sys.stderr = previous
    _controls_gone("\n".join(out))
    _controls_gone(err.getvalue())
    assert "pinball" in "\n".join(out)
    assert harness.calls


def test_flash_list_strips_entry_text(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash.jsonl")
    store.add(text=f"pinball {_ATTACK}", task="notes")
    out: list[str] = []
    concierge = _concierge(out, store=store)
    concierge.handle_line("/flash list")
    shown = "\n".join(out)
    _controls_gone(shown)
    assert "pinball" in shown


def test_job_summary_events_and_explain_are_sanitized_for_display() -> None:
    summary = f"pinball loss {_ATTACK}"
    result = SuperpowerResult(
        goal="g",
        plan=SuperpowerPlan(goal="g"),
        summary=summary,
        runs=[CapabilityRun("vol-bench", "vol-bench", "ran", f"tail {_ATTACK}")],
    )
    out: list[str] = []
    concierge = _concierge(out)
    job = BackgroundJob(label=f"job {_OSC_TITLE}", run_fn=lambda _cancel, _event: None)
    job.result = result
    job.events.append(("web", f"page {_ATTACK}"))
    job.done_event.set()
    concierge._job = job  # noqa: SLF001
    concierge.tick()
    shown = "\n".join(out)
    _controls_gone(shown)
    assert "pinball loss" in shown
    assert "tail" in shown
    assert concierge.last_answer == summary
    assert "\x1b]0;owned" in concierge.last_answer


def test_color_display_keeps_our_sgr_and_strips_osc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(concierge_module, "_stdout_color", lambda: True)
    out: list[str] = []
    concierge = _concierge(out)
    concierge._say(f"\x1b[2mnotice\x1b[0m {_ATTACK}")  # noqa: SLF001
    shown = out[0]
    assert "\x1b[2mnotice\x1b[0m" in shown
    assert "\x1b]" not in shown
    assert "owned" not in shown
    assert "\u202e" not in shown


def test_plain_display_strips_sgr_as_well(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(concierge_module, "_stdout_color", lambda: False)
    out: list[str] = []
    concierge = _concierge(out)
    concierge._say(f"\x1b[2mnotice\x1b[0m {_ATTACK}")  # noqa: SLF001
    assert out[0].startswith("notice")
    assert "\x1b" not in out[0]
    _controls_gone(out[0].replace("notice ", ""))


def test_line_separators_and_bidi_marks_do_not_survive_display() -> None:
    raw = "keep\u2028hide\u2029\u061c\u202eEND"
    cleaned = sanitize_for_terminal(raw)
    assert cleaned == "keephideEND"
    assert sanitize_for_terminal(cleaned) == cleaned
    out: list[str] = []
    _concierge(out)._say(raw)  # noqa: SLF001
    assert out == ["keephideEND"]


def test_console_error_paths_strip_controls() -> None:
    captured: list[str] = []
    _show_error(captured.append, RuntimeError(_ATTACK), to_output=True)
    _controls_gone(captured[0])
    assert captured[0].startswith("error:")
    err = io.StringIO()
    previous = sys.stderr
    sys.stderr = err
    try:
        _show_error(captured.append, RuntimeError(_ATTACK), to_output=False)
    finally:
        sys.stderr = previous
    _controls_gone(err.getvalue())


def test_writer_color_and_plain_strip_untrusted_controls() -> None:
    for color in (True, False):
        buf = io.StringIO()
        writer = ConsoleWriter(_caps(color=color), out=buf)
        writer.line(_ATTACK)
        writer.status(_ATTACK)
        text = buf.getvalue()
        assert "\x1b]" not in text
        assert "owned" not in text
        assert "\u202e" not in text
        assert "\u2028" not in text
    plain = io.StringIO()
    ConsoleWriter(_caps(color=False), out=plain).line("\x1b[31mred\x1b[0m")
    assert plain.getvalue().strip() == "red"
    colored = io.StringIO()
    writer = ConsoleWriter(_caps(color=True), out=colored)
    writer.line(Style(_caps(color=True)).dim("notice") + _OSC_TITLE)
    body = colored.getvalue()
    assert "\x1b[2mnotice\x1b[0m" in body
    assert "owned" not in body


def test_display_keeps_newlines_tabs_and_printable_unicode() -> None:
    text = "alpha\n\tbeta café 你好"
    assert sanitize_for_terminal(text) == text
    out: list[str] = []
    _concierge(out)._say(text)  # noqa: SLF001
    assert out == [text]


def test_cr_and_backspace_cannot_overwrite_a_trusted_prefix() -> None:
    raw = "trusted\rEVIL\x08"
    cleaned = sanitize_for_terminal(raw)
    assert cleaned == "trustedEVIL"
    assert "\r" not in cleaned
    assert "\x08" not in cleaned
    out: list[str] = []
    _concierge(out)._say(raw)  # noqa: SLF001
    assert out == ["trustedEVIL"]
