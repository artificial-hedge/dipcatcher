"""Chat console: the terminal engine wired into the concierge loop.

Injected ``input_fn`` / ``output_fn`` keep the historical plain console
(prompt ``dip › ``, ``startup_lines``, polling read). A real interactive
TTY, or ``engine=True``, uses :class:`LineReader`, the double Ctrl-C
policy, and a single status line while a background job runs.

Concierge methods run on the thread that called :func:`run_console`.
Cleanup order is shutdown, history save, telemetry close. Termios is
restored by :class:`LineReader` on every read, including interrupts.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import typer

from fx1.interactive.approvals import ApprovalGate, console_decide_fn
from fx1.interactive.model_ux import _endpoint_banner, public_error, startup_lines
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.profiles import DEFAULT_MODEL, MODELS, config_dir
from fx1.interactive.redact import redact_text
from fx1.interactive.runctx import is_valid_id, new_session_id
from fx1.interactive.telemetry import Telemetry
from fx1.interactive.terminal import (
    ConsoleWriter,
    InputCancelled,
    InterruptTracker,
    LineReader,
    TermCaps,
    detect_caps,
    format_duration,
)

_LEGACY_PROMPT = "dip › "
_HINT = "/help, /superpower, /exit"
_WINDOWS_NOTICE = "On Windows, background progress updates after you submit a line."
_SLASH_COMMANDS: tuple[str, ...] = (
    "/use",
    "/model",
    "/models",
    "/superpower",
    "/research",
    "/status",
    "/cancel",
    "/redirect",
    "/evidence",
    "/budget",
    "/session",
    "/sessions",
    "/resume",
    "/export",
    "/new",
    "/flash",
    "/harness",
    "/keys",
    "/approvals",
    "/diagnostics",
    "/help",
    "/exit",
)
_MISSING = object()


@dataclass(slots=True)
class _Progress:
    since: float | None = None


class _BoundInput:
    """Secondary prompts (approval, endpoint URL) share the line reader."""

    def __init__(self, fallback: Callable[[str], str]) -> None:
        self.fallback = fallback
        self.reader: LineReader | None = None

    def __call__(self, prompt: str) -> str:
        reader = self.reader
        if reader is not None and self.fallback is input:
            return reader.read_line(prompt)
        return self.fallback(prompt)


def completion_words() -> list[str]:
    """Slash commands, model ids, and harness command names, in that order."""
    words = list(_SLASH_COMMANDS)
    words.extend(MODELS)
    words.extend(_harness_names())
    return words


def _tty_polling_read(input_fn: Callable[[str], str], prompt: str, timeout_s: float) -> str | None:
    """Poll a POSIX TTY; Windows and non-TTY input use blocking reads.

    Returns None on timeout. The caller supplies an empty prompt while
    waiting for a previously prompted line.
    """
    if sys.platform == "win32" or not sys.stdin.isatty():
        return input_fn(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if not _stdin_ready(timeout_s):
        return None
    return _read_stdin_line()


def run_console(
    model: str = DEFAULT_MODEL,
    *,
    concierge_factory: Callable[..., Any] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = typer.echo,
    poll_timeout_s: float = 0.15,
    resume: bool = False,
    model_explicit: bool = False,
    engine: bool | None = None,
    caps_fn: Callable[[], TermCaps] | None = None,
    clock: Callable[[], float] | None = None,
    interrupt_clock: Callable[[], float] | None = None,
    secret_fn: Callable[[str], str] | None = None,
    telemetry_factory: Callable[[str], Telemetry] | None = None,
    history_path: Path | None = None,
    stream: TextIO | None = None,
) -> None:
    """Run the chat console until EOF, ``/exit``, or a second Ctrl-C.

    ``engine=None`` uses the line editor only when both streams are an
    interactive TTY and the caller left ``input_fn`` / ``output_fn`` at
    their defaults. ``engine=True`` forces the editor; injected callables
    still supply every byte and color stays off unless ``caps_fn`` says so.
    """
    provider = caps_fn if caps_fn is not None else detect_caps
    caps = provider()
    if _use_engine(engine, input_fn, output_fn, caps):
        _run_engine(
            model,
            caps=caps,
            caps_fn=provider,
            concierge_factory=concierge_factory,
            input_fn=input_fn,
            output_fn=output_fn,
            poll_timeout_s=poll_timeout_s,
            resume=resume,
            model_explicit=model_explicit,
            clock=_clock(clock),
            interrupt_clock=_clock(interrupt_clock),
            secret_fn=secret_fn,
            telemetry_factory=telemetry_factory,
            history_path=history_path,
            stream=stream,
        )
        return
    _run_legacy(
        model,
        concierge_factory=concierge_factory,
        input_fn=input_fn,
        output_fn=output_fn,
        poll_timeout_s=poll_timeout_s,
        resume=resume,
        model_explicit=model_explicit,
        telemetry_factory=telemetry_factory,
    )


def _run_engine(
    model: str,
    *,
    caps: TermCaps,
    caps_fn: Callable[[], TermCaps],
    concierge_factory: Callable[..., Any] | None,
    input_fn: Callable[[str], str],
    output_fn: Callable[..., None],
    poll_timeout_s: float,
    resume: bool,
    model_explicit: bool,
    clock: Callable[[], float],
    interrupt_clock: Callable[[], float],
    secret_fn: Callable[[str], str] | None,
    telemetry_factory: Callable[[str], Telemetry] | None,
    history_path: Path | None,
    stream: TextIO | None,
) -> None:
    prompt = _BoundInput(input_fn)
    concierge: Any = None
    tel: Telemetry | None = None
    reader: LineReader | None = None
    try:
        concierge, tel = _open_pair(
            model,
            concierge_factory,
            prompt,
            output_fn,
            resume=resume,
            model_explicit=model_explicit,
            animated=False,
            telemetry_factory=telemetry_factory,
        )
        _emit(tel, "console.start", _model_of(concierge, model))
        _emit_engine_banner(concierge, caps, output_fn)
        reader, writer = _open_reader(
            caps, caps_fn, clock, input_fn, secret_fn, history_path, stream
        )
        prompt.reader = reader
        _engine_loop(
            concierge,
            reader,
            writer,
            output_fn,
            InterruptTracker(monotonic=interrupt_clock),
            clock,
            poll_timeout_s,
            input_fn,
        )
    except (EOFError, KeyboardInterrupt):
        _blank(output_fn)
    finally:
        _finish(concierge, reader, tel, fallback_model=model)


def _run_legacy(
    model: str,
    *,
    concierge_factory: Callable[..., Any] | None,
    input_fn: Callable[[str], str],
    output_fn: Callable[..., None],
    poll_timeout_s: float,
    resume: bool,
    model_explicit: bool,
    telemetry_factory: Callable[[str], Telemetry] | None,
) -> None:
    concierge: Any = None
    tel: Telemetry | None = None
    try:
        concierge, tel = _open_pair(
            model,
            concierge_factory,
            input_fn,
            output_fn,
            resume=resume,
            model_explicit=model_explicit,
            animated=None,
            telemetry_factory=telemetry_factory,
        )
        _emit(tel, "console.start", _model_of(concierge, model))
        _legacy_banner(concierge, output_fn)
        _legacy_loop(concierge, input_fn, output_fn, poll_timeout_s)
    except (EOFError, KeyboardInterrupt):
        _blank(output_fn)
    finally:
        _finish(concierge, None, tel, fallback_model=model)


def _engine_loop(
    concierge: Any,
    reader: LineReader,
    writer: ConsoleWriter,
    output: Callable[..., None],
    tracker: InterruptTracker,
    clock: Callable[[], float],
    poll_timeout_s: float,
    input_fn: Callable[[str], str],
) -> None:
    progress = _Progress()
    while True:
        try:
            if _one_turn(
                concierge,
                reader,
                writer,
                output,
                tracker,
                clock,
                progress,
                poll_timeout_s,
                input_fn,
            ):
                return
        except InputCancelled:
            continue
        except EOFError:
            _blank(output)
            return
        except KeyboardInterrupt:
            if _on_prompt_interrupt(concierge, output, tracker):
                _blank(output)
                return
        except Exception as exc:  # noqa: BLE001 — one bad turn must not kill the shell
            _show_error(output, exc, to_output=True)


def _one_turn(
    concierge: Any,
    reader: LineReader,
    writer: ConsoleWriter,
    output: Callable[..., None],
    tracker: InterruptTracker,
    clock: Callable[[], float],
    progress: _Progress,
    poll_timeout_s: float,
    input_fn: Callable[[str], str],
) -> bool:
    writer.clear_status()
    line = reader.read_message(
        _engine_prompt(concierge),
        idle=_idle(concierge, writer, clock, progress),
        poll_s=poll_timeout_s,
        job_active=_job_active(concierge),
    )
    writer.clear_status()
    tracker.reset()
    ended = _submit_line(concierge, line, reader, output)
    _paint_after(concierge, writer, clock, progress, input_fn)
    return ended


def _on_prompt_interrupt(
    concierge: Any, output: Callable[..., None], tracker: InterruptTracker
) -> bool:
    """True when the session should exit. A running job is cancelled instead."""
    if _job_running(concierge):
        _request_job_cancel(concierge)
        output("cancel requested")
        return False
    if tracker.on_interrupt() == "hint":
        output(InterruptTracker.HINT)
        return False
    return True


def _submit_line(
    concierge: Any, line: str, reader: LineReader, output: Callable[..., None]
) -> bool:
    model = _keys_model(line, concierge)
    if model is None:
        return _dispatch_line(concierge, line, output, to_output=True)
    secret = _read_secret(reader, model, output)
    if secret is None:
        return False
    return _dispatch_with_secret(concierge, line, output, secret)


def _legacy_loop(
    concierge: Any,
    input_fn: Callable[[str], str],
    output: Callable[..., None],
    poll_timeout_s: float,
) -> None:
    pending = True
    while True:
        nxt = _legacy_step(concierge, input_fn, output, poll_timeout_s, pending)
        if nxt is None:
            return
        pending = nxt


def _legacy_step(
    concierge: Any,
    input_fn: Callable[[str], str],
    output: Callable[..., None],
    poll_timeout_s: float,
    pending: bool,
) -> bool | None:
    """Next prompt flag, or None when the session ended."""
    try:
        line = _tty_polling_read(input_fn, _legacy_prompt(pending), poll_timeout_s)
    except (EOFError, KeyboardInterrupt):
        _blank(output)
        return None
    return _legacy_after(concierge, line, output)


def _dispatch_line(
    concierge: Any, line: str, output: Callable[..., None], *, to_output: bool
) -> bool:
    try:
        return bool(concierge.handle_line(line))
    except KeyboardInterrupt:
        _cancel_turn(concierge)
        output("turn cancelled")
        return False
    except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the shell
        _show_error(output, exc, to_output=to_output)
        return False


def _dispatch_with_secret(
    concierge: Any, line: str, output: Callable[..., None], secret: str
) -> bool:
    with _one_shot_getpass(secret):
        return _dispatch_line(concierge, line, output, to_output=True)


def _legacy_prompt(pending: bool) -> str:
    if pending:
        return _LEGACY_PROMPT
    return ""


def _legacy_after(concierge: Any, line: str | None, output: Callable[..., None]) -> bool | None:
    if line is None:
        _call_tick(concierge)
        return False
    if _dispatch_line(concierge, line, output, to_output=False):
        return None
    _call_tick(concierge)
    return True


def _show_error(output: Callable[..., None], exc: Exception, *, to_output: bool) -> None:
    message = f"error: {public_error(exc)}"
    if to_output:
        output(message)
        return
    typer.secho(message, err=True, fg=typer.colors.RED)


def _emit_engine_banner(concierge: Any, caps: TermCaps, output: Callable[..., None]) -> None:
    model = _model_of(concierge, DEFAULT_MODEL)
    for line in _header_lines(model, caps):
        output(line)
    _emit_checks(concierge, output)
    output(_HINT)
    if sys.platform == "win32":
        output(_WINDOWS_NOTICE)
    _emit_notice(concierge, output)


def _legacy_banner(concierge: Any, output: Callable[..., None]) -> None:
    for line in startup_lines(_model_of(concierge, DEFAULT_MODEL)):
        output(line)
    _emit_notice(concierge, output)
    if sys.platform == "win32":
        output(_WINDOWS_NOTICE)


def _header_lines(model: str, caps: TermCaps) -> list[str]:
    host, source, extra = _endpoint_view(model)
    if caps.color:
        return [orb_banner(model, host, color=True), *extra]
    return [f"dipcatcher · {model} @ {host} · {source}", f"{model} @ {host}", *extra]


def _endpoint_view(model: str) -> tuple[str, str, list[str]]:
    host, extra = _endpoint_banner(model)
    return host, _source_from(extra), list(extra)


def _source_from(extra: list[str]) -> str:
    for line in extra:
        source = _source_token(line)
        if source:
            return source
    return "unset"


def _source_token(line: str) -> str:
    if not line.startswith("endpoint source "):
        return ""
    parts = line.split()
    if len(parts) < 3:
        return ""
    return parts[2]


def _emit_checks(concierge: Any, output: Callable[..., None]) -> None:
    method = getattr(concierge, "startup_checks", None)
    if not callable(method):
        _emit_store_health(output)
        return
    try:
        payload = method()
    except Exception as exc:  # noqa: BLE001 — startup text must not kill the shell
        _show_error(output, exc, to_output=True)
        return
    for line in _as_lines(payload):
        output(redact_text(line))


def _emit_store_health(output: Callable[..., None]) -> None:
    from fx1.interactive import profiles

    for item in profiles.store_health():
        output(f"warning: {item}")


def _emit_notice(concierge: Any, output: Callable[..., None]) -> None:
    notice = getattr(concierge, "resume_notice", "")
    if isinstance(notice, str) and notice:
        output(redact_text(notice))


def _open_pair(
    model: str,
    factory: Callable[..., Any] | None,
    prompt: Callable[[str], str],
    output_fn: Callable[..., None],
    *,
    resume: bool,
    model_explicit: bool,
    animated: bool | None,
    telemetry_factory: Callable[[str], Telemetry] | None,
) -> tuple[Any, Telemetry]:
    if factory is not None:
        return _open_factory(factory, telemetry_factory)
    return _open_default(
        model,
        prompt,
        output_fn,
        resume=resume,
        model_explicit=model_explicit,
        animated=animated,
        telemetry_factory=telemetry_factory,
    )


def _open_factory(
    factory: Callable[..., Any],
    telemetry_factory: Callable[[str], Telemetry] | None,
) -> tuple[Any, Telemetry]:
    concierge = factory()
    try:
        tel = _make_telemetry(_session_id(concierge), telemetry_factory)
    except Exception:  # noqa: BLE001 — construction failure still shuts the session down
        _shutdown(concierge)
        raise
    return concierge, tel


def _open_default(
    model: str,
    prompt: Callable[[str], str],
    output_fn: Callable[..., None],
    *,
    resume: bool,
    model_explicit: bool,
    animated: bool | None,
    telemetry_factory: Callable[[str], Telemetry] | None,
) -> tuple[Any, Telemetry]:
    tel = _make_telemetry(new_session_id(), telemetry_factory)
    try:
        concierge = _default_concierge(
            model=model,
            prompt=prompt,
            output_fn=output_fn,
            tel=tel,
            resume=resume,
            model_explicit=model_explicit,
            animated=animated,
        )
    except Exception:  # noqa: BLE001 — don't leave a telemetry sink open
        tel.close()
        raise
    return concierge, tel


def _default_concierge(
    *,
    model: str,
    prompt: Callable[[str], str],
    output_fn: Callable[..., None],
    tel: Telemetry,
    resume: bool,
    model_explicit: bool,
    animated: bool | None,
) -> Any:
    from fx1.interactive.concierge import Concierge

    store = _session_store()
    kwargs: dict[str, Any] = {
        "model": model,
        "input_fn": prompt,
        "output_fn": output_fn,
        "approvals": _approvals(prompt, output_fn),
        "session_store": store,
        "telemetry_emit": tel.as_emit(),
        "telemetry": tel,
    }
    if animated is not None:
        kwargs["animated"] = animated
    concierge = Concierge(**kwargs)
    _prepare_session(concierge, store, model=model, resume=resume, model_explicit=model_explicit)
    return concierge


def _prepare_session(
    concierge: Any,
    store: Any,
    *,
    model: str,
    resume: bool,
    model_explicit: bool,
) -> None:
    if store is None:
        return
    if resume:
        preferred = model if model_explicit else None
        concierge.restore_latest(preferred_model=preferred)
        return
    _mark_running_interrupted(concierge, store)


def _mark_running_interrupted(concierge: Any, store: Any) -> None:
    latest = store.latest()
    if not isinstance(latest, dict):
        return
    job = latest.get("job") or {}
    if not isinstance(job, dict) or job.get("state") != "running":
        return
    record = store.mark_interrupted(str(latest.get("id", "")))
    concierge.resume_notice = (
        f"session {record.get('id')} was interrupted and was not restarted. "
        "Use /resume to inspect it."
    )


def _session_store() -> Any | None:
    try:
        from fx1.interactive.sessions import SessionStore
    except ImportError:
        return None
    return SessionStore()


def _approvals(input_fn: Callable[[str], str], output_fn: Callable[..., None]) -> ApprovalGate:
    """Interactive consoles ask; piped sessions deny without reading stdin."""
    if not _stdin_is_tty():
        return ApprovalGate()
    return ApprovalGate(decide_fn=console_decide_fn(input_fn, _text_output(output_fn)))


def _text_output(output_fn: Callable[..., None]) -> Callable[[str], None]:
    def emit(text: str) -> None:
        output_fn(text)

    return emit


def _open_reader(
    caps: TermCaps,
    caps_fn: Callable[[], TermCaps],
    clock: Callable[[], float],
    input_fn: Callable[[str], str],
    secret_fn: Callable[[str], str] | None,
    history_path: Path | None,
    stream: TextIO | None,
) -> tuple[LineReader, ConsoleWriter]:
    writer = _writer(caps, caps_fn, clock, stream)
    reader = LineReader(
        caps,
        writer,
        history_path=_history_file(history_path),
        words=completion_words,
        input_fn=_reader_input(input_fn),
        secret_fn=secret_fn,
    )
    reader.load_history()
    return reader, writer


def _writer(
    caps: TermCaps,
    caps_fn: Callable[[], TermCaps],
    clock: Callable[[], float],
    stream: TextIO | None,
) -> ConsoleWriter:
    if stream is None:
        return ConsoleWriter(caps, caps_fn=caps_fn, clock=clock)
    return ConsoleWriter(caps, out=stream, caps_fn=caps_fn, clock=clock)


def _history_file(path: Path | None) -> Path:
    if path is not None:
        return path
    return config_dir() / "history"


def _reader_input(input_fn: Callable[[str], str]) -> Callable[[str], str] | None:
    if input_fn is input:
        return None
    return input_fn


def _finish(
    concierge: Any,
    reader: LineReader | None,
    tel: Telemetry | None,
    *,
    fallback_model: str,
) -> None:
    _shutdown(concierge)
    _close_reader(reader)
    if tel is not None and concierge is not None:
        _emit(tel, "console.exit", _model_of(concierge, fallback_model))
    if tel is not None:
        tel.close()


def _shutdown(concierge: Any) -> None:
    method = getattr(concierge, "shutdown", None)
    if not callable(method):
        return
    with contextlib.suppress(Exception):
        method()


def _close_reader(reader: LineReader | None) -> None:
    if reader is None:
        return
    with contextlib.suppress(Exception):
        reader.close()


def _make_telemetry(session_id: str, factory: Callable[[str], Telemetry] | None) -> Telemetry:
    if factory is not None:
        return factory(session_id)
    return Telemetry(session_id=session_id)


def _session_id(concierge: Any) -> str:
    exposed = getattr(concierge, "session_id", None)
    if isinstance(exposed, str) and is_valid_id(exposed, "s"):
        return exposed
    corr = getattr(concierge, "_corr_session", None)
    if isinstance(corr, str) and is_valid_id(corr, "s"):
        return corr
    return new_session_id()


def _emit(tel: Telemetry | None, event: str, model: str) -> None:
    if tel is None:
        return
    tel.emit(event, {"model": model})


def _model_of(concierge: Any, fallback: str) -> str:
    model = getattr(concierge, "model", fallback)
    if isinstance(model, str) and model:
        return model
    return fallback


def _engine_prompt(concierge: Any) -> str:
    return f"{_model_of(concierge, DEFAULT_MODEL)} › "


def _use_engine(
    engine: bool | None,
    input_fn: Callable[[str], str],
    output_fn: Callable[..., None],
    caps: TermCaps,
) -> bool:
    if engine is not None:
        return engine
    if input_fn is not input or output_fn is not typer.echo:
        return False
    return caps.interactive


def _idle(
    concierge: Any, writer: ConsoleWriter, clock: Callable[[], float], progress: _Progress
) -> Callable[[], None]:
    def callback() -> None:
        _paint_live(concierge, writer, clock, progress)

    return callback


def _paint_after(
    concierge: Any,
    writer: ConsoleWriter,
    clock: Callable[[], float],
    progress: _Progress,
    input_fn: Callable[[str], str],
) -> None:
    _drain_tick(concierge, writer.above_prompt)
    if not _job_running(concierge):
        progress.since = None
        return
    if _polls(writer.caps, input_fn):
        return
    writer.above_prompt(_status_text(concierge, _elapsed(progress, clock)))


def _paint_live(
    concierge: Any, writer: ConsoleWriter, clock: Callable[[], float], progress: _Progress
) -> None:
    _drain_tick(concierge, writer.above_prompt)
    if not _job_running(concierge):
        progress.since = None
        return
    text = _status_text(concierge, _elapsed(progress, clock))
    if writer.caps.interactive:
        writer.status(text)
        return
    writer.above_prompt(text)


def _elapsed(progress: _Progress, clock: Callable[[], float]) -> float:
    if progress.since is None:
        progress.since = clock()
    return clock() - progress.since


def _status_text(concierge: Any, elapsed: float) -> str:
    return f"{_job_label(concierge)} {format_duration(elapsed)}"


def _job_label(concierge: Any) -> str:
    named = getattr(concierge, "job_label", None)
    if isinstance(named, str) and named.strip():
        return named.strip()
    job = getattr(concierge, "_job", None)
    label = getattr(job, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return "job"


def _drain_tick(concierge: Any, emit: Callable[[str], None]) -> None:
    method = getattr(concierge, "tick", None)
    if not callable(method):
        return
    original = getattr(concierge, "_output", _MISSING)
    if not callable(original):
        method()
        return
    captured: list[str] = []
    concierge._output = _collector(captured)
    try:
        method()
    finally:
        concierge._output = original
    for line in captured:
        emit(redact_text(line))


def _collector(sink: list[str]) -> Callable[..., None]:
    def collect(text: str = "", *_args: Any, **_kwargs: Any) -> None:
        sink.append(str(text))

    return collect


def _job_active(concierge: Any) -> Callable[[], bool]:
    def active() -> bool:
        return _job_running(concierge)

    return active


def _job_running(concierge: Any) -> bool:
    value = getattr(concierge, "job_running", _MISSING)
    if value is not _MISSING:
        return bool(value)
    job = getattr(concierge, "_job", None)
    return bool(getattr(job, "running", False))


def _request_job_cancel(concierge: Any) -> None:
    try:
        _invoke_job_cancel(concierge)
    except Exception:  # noqa: BLE001 — cancel must not take down the prompt
        return


def _invoke_job_cancel(concierge: Any) -> None:
    method = getattr(concierge, "cancel_job", None)
    if callable(method):
        method()
        return
    job = getattr(concierge, "_job", None)
    cancel = getattr(job, "cancel", None)
    if callable(cancel):
        cancel()


def _cancel_turn(concierge: Any) -> None:
    method = getattr(concierge, "cancel_turn", None)
    if callable(method):
        method()


def _call_tick(concierge: Any) -> None:
    method = getattr(concierge, "tick", None)
    if callable(method):
        method()


def _keys_model(line: str, concierge: Any) -> str | None:
    tokens = _split_command(line)
    if tokens is None or len(tokens) < 2 or tokens[0] != "/keys" or tokens[1] != "set":
        return None
    if len(tokens) > 3:
        return None
    if len(tokens) == 3:
        return tokens[2]
    return _model_of(concierge, DEFAULT_MODEL)


def _split_command(line: str) -> list[str] | None:
    try:
        return shlex.split(line.strip())
    except ValueError:
        return None


def _read_secret(reader: LineReader, model: str, output: Callable[..., None]) -> str | None:
    try:
        return reader.read_secret(f"› {model} API key: ")
    except (EOFError, KeyboardInterrupt):
        output(f"endpoint setup for {model!r} cancelled; no changes saved")
        return None


def _polls(caps: TermCaps, input_fn: Callable[[str], str]) -> bool:
    if input_fn is not input or caps.windows or os.name != "posix":
        return False
    return caps.stdin_tty


def _as_lines(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _harness_names() -> list[str]:
    try:
        from fx1.harness import HARNESS_REGISTRY
    except ImportError:
        return []
    return [str(command.name) for command in HARNESS_REGISTRY]


def _stdin_ready(timeout_s: float) -> bool:
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    return bool(ready)


def _read_stdin_line() -> str:
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _clock(clock: Callable[[], float] | None) -> Callable[[], float]:
    if clock is None:
        return time.monotonic
    return clock


def _blank(output: Callable[..., None]) -> None:
    with contextlib.suppress(Exception):
        output("")


@contextlib.contextmanager
def _one_shot_getpass(secret: str) -> Iterator[None]:
    import getpass

    original = getpass.getpass

    def once(prompt: str = "", stream: TextIO | None = None) -> str:
        del prompt, stream
        return secret

    getpass.getpass = once
    try:
        yield
    finally:
        getpass.getpass = original


__all__ = [
    "completion_words",
    "run_console",
    "_tty_polling_read",
]
