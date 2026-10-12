"""Unit tests for the interactive console. Injected seams, no real TTY."""

from __future__ import annotations

import io
import json
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import concierge as concierge_module
from fx1.interactive import console
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.profiles import fingerprint
from fx1.interactive.telemetry import Telemetry
from fx1.interactive.terminal import InterruptTracker, TermCaps

_KEY = "sk-test-secret-value"


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Job:
    def __init__(self) -> None:
        self.running = True
        self.label = "research"
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.running = False


class Fake:
    """Minimal concierge seam for the console loop."""

    def __init__(self, model: str = "fx1") -> None:
        self.model = model
        self.resume_notice = ""
        self.lines: list[str] = []
        self.shut = False
        self.job_running = False
        self.job_label = "job"
        self.cancelled_job = False
        self.cancelled_turn = False
        self.fail = False
        self.interrupt = False
        self.checks: list[str] | None = []
        self.check_error: Exception | None = None
        self._job: _Job | None = None
        self.notes: list[str] = []
        self._output: Callable[..., None] = self.notes.append
        self.session_id = ""

    def startup_checks(self) -> list[str]:
        if self.check_error is not None:
            raise self.check_error
        if self.checks is None:
            raise AttributeError
        return list(self.checks)

    def handle_line(self, line: str) -> bool:
        self.lines.append(line)
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if self.fail:
            self.fail = False
            raise RuntimeError(f"boom {_KEY}")
        if line.strip() == "start":
            self.job_running = True
        if line.strip() in {"/exit", "exit"}:
            self.shutdown()
            return True
        return False

    def tick(self) -> None:
        if self.job_running:
            self._output("[job] working: step")

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        del timeout_s
        self.shut = True
        return True

    def cancel_turn(self) -> None:
        self.cancelled_turn = True

    def cancel_job(self) -> bool:
        self.cancelled_job = True
        self.job_running = False
        return True


def _caps(**overrides: object) -> TermCaps:
    values: dict[str, object] = {
        "stdin_tty": False,
        "stdout_tty": False,
        "interactive": False,
        "color": False,
        "unicode": True,
        "width": 80,
        "height": 24,
        "windows": False,
    }
    values.update(overrides)
    return TermCaps(
        stdin_tty=bool(values["stdin_tty"]),
        stdout_tty=bool(values["stdout_tty"]),
        interactive=bool(values["interactive"]),
        color=bool(values["color"]),
        unicode=bool(values["unicode"]),
        width=int(values["width"]),  # type: ignore[arg-type]
        height=int(values["height"]),  # type: ignore[arg-type]
        windows=bool(values["windows"]),
    )


def _script(lines: list[str]) -> Callable[[str], str]:
    pending = list(lines)

    def read(prompt: str) -> str:
        del prompt
        if not pending:
            raise EOFError
        return pending.pop(0)

    return read


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return cfg


def _run(fake: Fake, read: Callable[[str], str], out: list[str], **kwargs: Any) -> None:
    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=out.append,
        concierge_factory=lambda: fake,
        caps_fn=kwargs.pop("caps_fn", lambda: _caps()),
        **kwargs,
    )


@pytest.mark.parametrize("model", ["fx1", "fx1-lite"])
def test_prompt_uses_the_active_model(isolated: Path, model: str) -> None:
    del isolated
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    _run(Fake(model), read, [])
    assert prompts == [f"{model} › "]


def test_prompt_updates_after_use(isolated: Path) -> None:
    out: list[str] = []
    prompts: list[str] = []
    queued = ["/use fx1-lite", "/exit"]

    def read(prompt: str) -> str:
        prompts.append(prompt)
        if not queued:
            raise EOFError
        return queued.pop(0)

    concierge = Concierge(
        animated=False,
        approvals=ApprovalGate(),
        session_store=None,
        output_fn=out.append,
    )
    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=out.append,
        concierge_factory=lambda: concierge,
        caps_fn=lambda: _caps(),
    )
    assert prompts == ["fx1 › ", "fx1-lite › "]
    assert concierge.model == "fx1-lite"


def test_startup_checks_print_once_before_the_hint(isolated: Path) -> None:
    del isolated
    fake = Fake()
    fake.checks = ["store healthy", "job was interrupted"]
    out: list[str] = []
    _run(fake, _script([]), out)
    text = _joined(out)
    assert text.count("store healthy") == 1
    assert text.index("store healthy") < text.index("/help, /superpower, /exit")
    assert "job was interrupted" in text


def test_startup_check_errors_are_redacted_and_the_loop_continues(isolated: Path) -> None:
    del isolated
    fake = Fake()
    fake.check_error = RuntimeError(f"leaked {_KEY}")
    out: list[str] = []
    _run(fake, _script(["/exit"]), out)
    text = _joined(out)
    assert "error:" in text
    assert "[REDACTED]" in text
    assert _KEY not in text
    assert fake.shut
    assert fake.lines == ["/exit"]


def test_missing_startup_checks_fall_back_to_store_health(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated
    monkeypatch.setattr("fx1.interactive.profiles.store_health", lambda: ["disk warning"])
    fake = Fake()
    fake.checks = None
    out: list[str] = []

    def read(prompt: str) -> str:
        del prompt
        raise EOFError

    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=out.append,
        concierge_factory=lambda: _NoChecks(),
        caps_fn=lambda: _caps(),
    )
    assert "warning: disk warning" in out


class _NoChecks:
    model = "fx1"
    resume_notice = ""

    def handle_line(self, line: str) -> bool:
        del line
        return False

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        del timeout_s
        return True


def test_plain_banner_names_the_missing_endpoint(isolated: Path) -> None:
    del isolated
    out: list[str] = []
    _run(Fake(), _script([]), out)
    text = _joined(out)
    assert "dipcatcher · fx1 @ unconfigured · unset" in text
    assert "fx1 @ unconfigured" in text
    assert "No endpoint configured. Run /keys set fx1" in text
    assert "\x1b" not in text


def test_plain_banner_shows_host_source_and_fingerprint(isolated: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", _KEY, "https://fx1.example.test/v1")
    out: list[str] = []
    _run(Fake(), _script([]), out)
    text = _joined(out)
    assert "dipcatcher · fx1 @ fx1.example.test · stored" in text
    assert fingerprint(_KEY) in text
    assert _KEY not in text
    assert "\x1b" not in text


def test_invalid_stored_url_is_not_echoed(isolated: Path) -> None:
    from fx1.interactive import profiles

    path = profiles.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "profiles": {
                    "fx1": {
                        "api_key": _KEY,
                        "base_url": "file:///etc/passwd",
                        "service_origin": "file://",
                        "set_at": "2026-01-01T00:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    out: list[str] = []
    _run(Fake(), _script([]), out)
    text = _joined(out)
    assert _KEY not in text
    assert "file://" not in text
    assert "invalid" in text.casefold()


def test_windows_notice_and_no_select(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del isolated

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("select")

    monkeypatch.setattr("select.select", boom)
    monkeypatch.setattr(console.sys, "platform", "win32")
    out: list[str] = []
    _run(Fake(), _script(["/model"]), out, caps_fn=lambda: _caps(windows=True))
    assert "background progress updates after you submit a line" in _joined(out)


def test_color_banner_contains_an_escape(isolated: Path) -> None:
    del isolated
    out: list[str] = []
    _run(Fake(), _script([]), out, caps_fn=lambda: _caps(color=True, interactive=True))
    text = _joined(out)
    assert "\x1b" in text
    assert "fx1" in text


def test_no_color_beats_force_color(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    out: list[str] = []
    console.run_console(engine=True, input_fn=_script([]), output_fn=out.append)
    text = _joined(out)
    assert "\x1b" not in text
    assert "dipcatcher · fx1 @" in text


def test_first_ctrl_c_hints_and_second_inside_the_window_exits(isolated: Path) -> None:
    del isolated
    clock = _Clock()
    fake = Fake()
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 2:
            clock.now = 1.0
        if seen["n"] >= 3:
            raise AssertionError("session kept running")
        raise KeyboardInterrupt

    out: list[str] = []
    _run(fake, read, out, interrupt_clock=clock)
    assert InterruptTracker.HINT in out
    assert fake.shut
    assert seen["n"] == 2


def test_ctrl_c_outside_the_window_hints_again(isolated: Path) -> None:
    del isolated
    clock = _Clock()
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 2:
            clock.now = 5.0
            raise KeyboardInterrupt
        if seen["n"] > 2:
            raise EOFError
        raise KeyboardInterrupt

    out: list[str] = []
    _run(Fake(), read, out, interrupt_clock=clock)
    assert out.count(InterruptTracker.HINT) == 2
    assert seen["n"] == 3


def test_a_submitted_line_resets_the_exit_window(isolated: Path) -> None:
    del isolated
    clock = _Clock()
    fake = Fake()
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 1:
            raise KeyboardInterrupt
        if seen["n"] == 2:
            return "hello"
        if seen["n"] == 3:
            clock.now = 0.5
            raise KeyboardInterrupt
        raise EOFError

    out: list[str] = []
    _run(fake, read, out, interrupt_clock=clock)
    assert out.count(InterruptTracker.HINT) == 2
    assert "hello" in fake.lines


def test_ctrl_c_while_a_job_runs_requests_cancel(isolated: Path) -> None:
    del isolated
    fake = Fake()
    fake.job_running = True
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 1:
            raise KeyboardInterrupt
        raise EOFError

    out: list[str] = []
    _run(fake, read, out)
    assert "cancel requested" in out
    assert fake.cancelled_job
    assert not fake.job_running
    assert seen["n"] == 2
    assert fake.shut


def test_cancel_falls_back_to_the_job_object(isolated: Path) -> None:
    del isolated

    class Bare:
        model = "fx1"
        resume_notice = ""

        def __init__(self) -> None:
            self._job = _Job()
            self.shut = False

        def handle_line(self, line: str) -> bool:
            del line
            return False

        def shutdown(self, timeout_s: float = 5.0) -> bool:
            del timeout_s
            self.shut = True
            return True

    bare = Bare()
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 1:
            raise KeyboardInterrupt
        raise EOFError

    out: list[str] = []
    _run(bare, read, out)  # type: ignore[arg-type]
    assert bare._job.cancelled
    assert "cancel requested" in out


def test_eof_shuts_down_saves_history_and_closes_telemetry(isolated: Path) -> None:
    opened: list[Telemetry] = []

    def factory(session_id: str) -> Telemetry:
        tel = Telemetry(session_id=session_id, directory=isolated / "logs")
        opened.append(tel)
        return tel

    fake = Fake()
    _run(fake, _script(["hello"]), [], telemetry_factory=factory)
    assert fake.shut
    body = json.loads((isolated / "history").read_text(encoding="utf-8"))
    assert body == ["hello"]
    assert stat.S_IMODE((isolated / "history").stat().st_mode) == 0o600
    assert opened
    tel = opened[0]
    before = tel.path.read_text(encoding="utf-8")
    assert "console.start" in before
    tel.emit("console.after", {"model": "fx1"})
    assert tel.path.read_text(encoding="utf-8") == before


def test_exit_command_shuts_down(isolated: Path) -> None:
    del isolated
    fake = Fake()
    _run(fake, _script(["/exit"]), [])
    assert fake.lines == ["/exit"]
    assert fake.shut


def test_empty_line_does_not_leave(isolated: Path) -> None:
    del isolated
    fake = Fake()
    _run(fake, _script(["", "/exit"]), [])
    assert fake.lines == ["", "/exit"]


def test_multiline_continuation_is_one_submission(isolated: Path) -> None:
    del isolated
    fake = Fake()
    _run(fake, _script(["one\\", "two", "/exit"]), [])
    assert fake.lines[0] == "one\ntwo"


def test_fenced_block_drops_the_fences(isolated: Path) -> None:
    del isolated
    fake = Fake()
    _run(fake, _script(["```", "alpha", "```", "/exit"]), [])
    assert fake.lines[0] == "alpha"


def test_ctrl_d_inside_a_fence_discards_the_block(isolated: Path) -> None:
    del isolated
    fake = Fake()
    seen = {"n": 0}

    def read(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 1:
            return '"""'
        if seen["n"] == 2:
            raise EOFError
        return "/exit"

    _run(fake, read, [])
    assert fake.lines == ["/exit"]
    assert fake.shut


def test_turn_interrupt_on_a_fake_prints_turn_cancelled(isolated: Path) -> None:
    del isolated
    fake = Fake()
    fake.interrupt = True
    out: list[str] = []
    _run(fake, _script(["slow", "/exit"]), out)
    assert "turn cancelled" in out
    assert fake.cancelled_turn
    assert fake.lines == ["slow", "/exit"]


def test_real_turn_interrupt_is_not_printed_twice(isolated: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", _KEY, "https://fx1.example.test/v1")

    class Backend:
        def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            del messages, kwargs
            raise KeyboardInterrupt

    cout: list[str] = []
    sout: list[str] = []
    concierge = Concierge(
        animated=False,
        approvals=ApprovalGate(),
        session_store=None,
        output_fn=cout.append,
        backend_factory=lambda **kwargs: Backend(),
    )
    console.run_console(
        engine=True,
        input_fn=_script(["hello", "/exit"]),
        output_fn=sout.append,
        concierge_factory=lambda: concierge,
        caps_fn=lambda: _caps(),
    )
    assert cout.count("turn cancelled") == 1
    assert "turn cancelled" not in _joined(sout)
    assert _KEY not in _joined(cout + sout)


def test_a_failing_turn_stays_in_the_loop_and_hides_the_key(isolated: Path) -> None:
    del isolated
    fake = Fake()
    fake.fail = True
    out: list[str] = []
    _run(fake, _script(["bad", "/exit"]), out)
    text = _joined(out)
    assert "error:" in text
    assert "[REDACTED]" in text
    assert _KEY not in text
    assert fake.lines == ["bad", "/exit"]


def test_keys_set_uses_secret_fn_and_does_not_record_the_key(isolated: Path) -> None:
    prompts: list[str] = []
    secrets: list[str] = []
    queued = ["/keys set fx1", "https://example.test/v1"]

    def read(prompt: str) -> str:
        prompts.append(prompt)
        if not queued:
            raise EOFError
        return queued.pop(0)

    def secret(prompt: str) -> str:
        secrets.append(prompt)
        return _KEY

    out: list[str] = []
    concierge = Concierge(
        animated=False,
        approvals=ApprovalGate(),
        session_store=None,
        input_fn=read,
        output_fn=out.append,
    )
    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=out.append,
        concierge_factory=lambda: concierge,
        caps_fn=lambda: _caps(),
        secret_fn=secret,
    )
    text = _joined(out)
    history = (isolated / "history").read_text(encoding="utf-8")
    assert secrets == ["› fx1 API key: "]
    assert "stored" in text
    assert _KEY not in text
    assert _KEY not in history
    assert "/keys" not in history


def test_history_keeps_chat_and_skips_keys_lines(isolated: Path) -> None:
    fake = Fake()
    _run(fake, _script(["hello", "/keys list"]), [])
    body = json.loads((isolated / "history").read_text(encoding="utf-8"))
    assert body == ["hello"]
    assert "/keys" not in json.dumps(body)


def test_completion_words_cover_commands_models_and_harness() -> None:
    words = console.completion_words()
    assert words[0] == "/use"
    assert "/help" in words and "/diagnostics" in words and "/sessions" in words
    assert "fx1" in words and "fx1-lite" in words
    assert "doctor" in words
    assert words.index("/exit") < words.index("fx1") < words.index("doctor")


def test_caps_refresh_on_each_prompt(isolated: Path) -> None:
    del isolated
    seen = {"n": 0}

    def caps_fn() -> TermCaps:
        seen["n"] += 1
        return _caps(width=40 + seen["n"])

    _run(Fake(), _script(["a", "b"]), [], caps_fn=caps_fn)
    assert seen["n"] >= 3


def test_background_progress_is_a_plain_status_line(isolated: Path) -> None:
    del isolated
    clock = _Clock()
    fake = Fake()
    fake.job_label = "research"
    buf = io.StringIO()
    queued = ["start", "stay"]

    def read(prompt: str) -> str:
        del prompt
        if not queued:
            raise EOFError
        line = queued.pop(0)
        if line == "stay":
            clock.now = 2.5
        return line

    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=lambda text: None,
        concierge_factory=lambda: fake,
        caps_fn=lambda: _caps(),
        clock=clock,
        stream=buf,
    )
    text = buf.getvalue()
    assert "[job] working: step" in text
    assert "research 0ms" in text
    assert "research 2.5s" in text
    assert "\x1b" not in text


def test_telemetry_uses_an_exposed_session_id(isolated: Path) -> None:
    opened: list[str] = []

    def factory(session_id: str) -> Telemetry:
        opened.append(session_id)
        return Telemetry(session_id=session_id, directory=isolated / "logs")

    fake = Fake()
    fake.session_id = "s-20261012T000000Z-abcdef01"
    _run(fake, _script([]), [], telemetry_factory=factory)
    assert opened == ["s-20261012T000000Z-abcdef01"]
    assert (isolated / "logs" / "s-20261012T000000Z-abcdef01.jsonl").is_file()


def test_a_hostile_session_id_is_not_a_path(isolated: Path) -> None:
    opened: list[str] = []

    def factory(session_id: str) -> Telemetry:
        opened.append(session_id)
        return Telemetry(session_id=session_id, directory=isolated / "logs")

    fake = Fake()
    fake.session_id = "../etc/passwd"
    _run(fake, _script([]), [], telemetry_factory=factory)
    assert opened and ".." not in opened[0]
    assert opened[0].startswith("s-")


def test_log_off_suppresses_console_events(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_LOG", "off")
    console.run_console(input_fn=_script([]), output_fn=lambda text: None)
    logs = list((isolated / "logs").glob("*.jsonl"))
    blob = "".join(path.read_text(encoding="utf-8") for path in logs)
    assert "console.start" not in blob


def test_default_factory_emits_console_start(isolated: Path) -> None:
    out: list[str] = []
    console.run_console(input_fn=_script([]), output_fn=out.append)
    logs = list((isolated / "logs").glob("*.jsonl"))
    assert logs
    body = logs[0].read_text(encoding="utf-8")
    assert "console.start" in body
    assert "console.exit" in body
    assert _KEY not in body
    assert "dipcatcher concierge" in _joined(out)


def test_resume_prints_the_saved_session(isolated: Path) -> None:
    from fx1.interactive.sessions import SessionStore

    store = SessionStore()
    record = store.create(model="fx1", messages=[{"role": "user", "content": "hi"}])
    out: list[str] = []
    console.run_console(resume=True, input_fn=_script([]), output_fn=out.append)
    assert f"resumed session {record['id']}" in _joined(out)


def test_model_explicit_resume_keeps_the_requested_model(isolated: Path) -> None:
    from fx1.interactive.sessions import SessionStore

    SessionStore().create(model="fx1", messages=[])
    out: list[str] = []
    console.run_console(
        model="fx1-lite",
        resume=True,
        model_explicit=True,
        input_fn=_script([]),
        output_fn=out.append,
    )
    assert "on model fx1-lite" in _joined(out)


def test_resume_without_explicit_model_keeps_the_stored_one(isolated: Path) -> None:
    from fx1.interactive.sessions import SessionStore

    SessionStore().create(model="fx1", messages=[])
    out: list[str] = []
    console.run_console(
        model="fx1-lite",
        resume=True,
        model_explicit=False,
        input_fn=_script([]),
        output_fn=out.append,
    )
    assert "on model fx1" in _joined(out)


def test_a_running_session_is_marked_interrupted(isolated: Path) -> None:
    from fx1.interactive.sessions import SessionStore

    store = SessionStore()
    record = store.create(model="fx1", messages=[])
    record["job"]["state"] = "running"
    store.save(record)
    out: list[str] = []
    console.run_console(input_fn=_script([]), output_fn=out.append)
    assert "not restarted" in _joined(out)
    loaded = store.load(str(record["id"]))
    assert loaded["job"]["state"] == "interrupted"


def test_injected_seams_keep_the_legacy_prompt_and_no_ansi(isolated: Path) -> None:
    del isolated
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    out: list[str] = []
    console.run_console(input_fn=read, output_fn=out.append, concierge_factory=lambda: Fake())
    assert prompts == ["dip › "]
    assert "dipcatcher concierge" in _joined(out)
    assert "\x1b" not in _joined(out)


def test_wrapper_forwards_engine_and_legacy(isolated: Path) -> None:
    del isolated
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    concierge_module.run_console(
        engine=True,
        input_fn=read,
        output_fn=lambda text: None,
        concierge_factory=lambda: Fake(),
        caps_fn=lambda: _caps(),
    )
    assert prompts == ["fx1 › "]
    prompts.clear()

    def legacy(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    out: list[str] = []
    concierge_module.run_console(
        input_fn=legacy, output_fn=out.append, concierge_factory=lambda: Fake()
    )
    assert prompts == ["dip › "]
    assert "dipcatcher concierge" in _joined(out)


def test_polling_read_uses_input_fn_off_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(console.sys, "stdin", io.StringIO("ignored\n"))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("select")

    monkeypatch.setattr("select.select", boom)
    assert console._tty_polling_read(lambda prompt: "x", "p> ", 0.01) == "x"


def test_polling_read_on_windows_does_not_select(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(console.sys, "stdin", Tty())
    monkeypatch.setattr(console.sys, "platform", "win32")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("select")

    monkeypatch.setattr("select.select", boom)
    seen: list[str] = []

    def read(prompt: str) -> str:
        seen.append(prompt)
        return "win"

    assert console._tty_polling_read(read, "dip › ", 0.01) == "win"
    assert seen == ["dip › "]


def test_polling_read_propagates_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(console.sys, "stdin", io.StringIO(""))

    def read(prompt: str) -> str:
        del prompt
        raise EOFError

    with pytest.raises(EOFError):
        console._tty_polling_read(read, "p> ", 0.01)


def test_wrapper_polling_read_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(concierge_module.sys, "stdin", io.StringIO(""))
    assert concierge_module._tty_polling_read(lambda prompt: "z", "q ", 0.01) == "z"


def test_hint_is_the_tracker_sentence(isolated: Path) -> None:
    del isolated
    seen = {"n": 0}

    def once(prompt: str) -> str:
        del prompt
        seen["n"] += 1
        if seen["n"] == 1:
            raise KeyboardInterrupt
        raise EOFError

    out: list[str] = []
    _run(Fake(), once, out)
    assert InterruptTracker.HINT in out


def test_factory_lines_reach_the_concierge(isolated: Path) -> None:
    del isolated
    fake = Fake()
    _run(fake, _script(["plain", "/exit"]), [])
    assert fake.lines == ["plain", "/exit"]
