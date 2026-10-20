"""TTY console driver for the interactive concierge."""

from __future__ import annotations

import select
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import typer

from fx1.interactive import lineio, profiles
from fx1.interactive import session as session_mod
from fx1.interactive.approvals import ApprovalGate, console_ask_fn
from fx1.interactive.concierge_slash import ACTIVE_JOB_POLL_S
from fx1.interactive.endpoint import explain_backend_error, format_guidance
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.orb import orbs_enabled
from fx1.interactive.profiles import DEFAULT_MODEL

if TYPE_CHECKING:
    from fx1.interactive.concierge import Concierge


class _null:
    """Context manager that does nothing (animation disabled / non-TTY)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _BlockingReader:
    """Blocking fallback when the platform offers no pollable console input."""

    mode = "blocking"
    interactive = False

    def __init__(self, input_fn: Callable[[str], str]) -> None:
        self._input = input_fn

    def read(self, prompt: str, timeout_s: float) -> str | None:
        del timeout_s
        return self._input(prompt)


def _line_reader(input_fn: Callable[[str], str]) -> lineio.LineReader | _BlockingReader:
    """POSIX TTYs poll with select; Windows consoles poll with msvcrt; a Windows
    build without msvcrt (or any non-TTY stream) falls back to blocking reads."""
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return _BlockingReader(input_fn)
        if not sys.stdin.isatty():
            return _BlockingReader(input_fn)
        return lineio.LineReader(
            input_fn=input_fn,
            platform="win32",
            kbhit=msvcrt.kbhit,
            getwch=msvcrt.getwch,
        )
    if not sys.stdin.isatty():
        return _BlockingReader(input_fn)
    return lineio.LineReader(input_fn=input_fn, platform="linux", select_fn=select.select)


def _tty_polling_read(input_fn: Callable[[str], str], prompt: str, timeout_s: float) -> str | None:
    """Poll a POSIX TTY; Windows and non-TTY input use blocking reads.

    Kept for callers of the pre-``LineReader`` API. Returns None on timeout.
    """
    return _line_reader(input_fn).read(prompt, timeout_s)


def _print_startup(
    concierge: Concierge, output_fn: Callable[..., None], resume: str | None
) -> None:
    """Banner, endpoint status, budget, interrupted-job reconciliation, optional resume."""
    model = concierge.model
    try:
        endpoint = profiles.resolve_endpoint(model)
        endpoint_problem: str | None = None
    except ValueError as exc:
        endpoint = None
        endpoint_problem = format_guidance(explain_backend_error(exc, model=model))
    host = profiles.host_of(endpoint[1]) if endpoint else "unconfigured"
    output_fn(orb_banner(model, host, color=orbs_enabled()))
    if endpoint is not None:
        from fx1.interactive.model_ux import _endpoint_banner

        _banner_host, extra = _endpoint_banner(model)
        del _banner_host
        for line in extra:
            output_fn(line)
    output_fn(
        f"dipcatcher concierge — chat id {model} (no released weights). Plain text chats; "
        "/superpower, /research, /flash, /help. Ctrl-D or /exit to leave."
    )
    output_fn(
        f"session {concierge.session_id} · budget {concierge.run_budget.describe()} · "
        "Ctrl-C cancels the running job; twice exits"
    )
    for note in concierge.startup_notes:
        output_fn(f"note: {note}")
    if endpoint_problem is not None:
        output_fn(endpoint_problem)
    elif endpoint is None:
        output_fn(
            f"No endpoint configured. Run /keys set {model} to add your API key and URL, "
            "or /doctor for a readiness checklist."
        )
    try:
        interrupted = session_mod.reconcile_interrupted(exclude_session_id=concierge.session_id)
    except Exception as exc:  # noqa: BLE001 — a damaged journal dir must not block startup
        interrupted = []
        output_fn(
            f"note: could not reconcile prior session journals ({type(exc).__name__}: {exc}); "
            "new work continues — /sessions may be incomplete until the journal is repaired"
        )
    if interrupted:
        ids = ", ".join(sorted({sid for sid, _ in interrupted}))
        output_fn(
            f"{len(interrupted)} job(s) from earlier sessions were still open and are now "
            f"marked interrupted (nothing is re-run): {ids} — /sessions, /resume <id>"
        )
    if resume:
        try:
            concierge.handle_line(f"/resume {resume}")
        except Exception as exc:  # noqa: BLE001
            output_fn(f"could not resume {resume!r}: {exc}")


def _on_interrupt(
    concierge: Concierge, interrupts: lineio.InterruptPolicy, output_fn: Callable[..., None]
) -> bool:
    """Apply the Ctrl-C policy; True when the console should exit."""
    job = concierge.job
    action = interrupts.on_interrupt(job_running=bool(job and job.running))
    output_fn("")
    if action == lineio.InterruptAction.EXIT:
        concierge.shutdown()
        return True
    if action == lineio.InterruptAction.CANCEL_JOB:
        concierge.cancel_job()
    output_fn(interrupts.hint(action))
    return False


def _dispatch_line(concierge: Concierge, line: str, output_fn: Callable[..., None]) -> bool:
    """Run one submitted line; True when the session should end."""
    try:
        return concierge.handle_line(line)
    except KeyboardInterrupt:
        concierge.cancel_job()
        output_fn("")
        output_fn("turn interrupted — the running job (if any) was asked to stop")
    except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the shell
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
    return False


def _console_poll_timeout(concierge: Concierge, idle_timeout_s: float) -> float:
    job = concierge.job
    if job is not None and job.running:
        return ACTIVE_JOB_POLL_S
    return idle_timeout_s


def run_console(
    model: str = DEFAULT_MODEL,
    *,
    concierge_factory: Callable[..., Any] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = typer.echo,
    poll_timeout_s: float = 0.15,
    resume: str | None = None,
) -> None:
    """Read chat input with idle progress ticks on POSIX terminals.

    Windows terminals poll the console; non-TTY input renders queued progress
    between submitted lines. Ctrl-C once cancels the running job (or clears
    the line); twice within two seconds exits.
    """
    from fx1.interactive.concierge import Concierge

    concierge = (
        concierge_factory()
        if concierge_factory is not None
        else Concierge(
            model=model,
            input_fn=input_fn,
            output_fn=output_fn,
            approvals=ApprovalGate(ask_fn=console_ask_fn(input_fn)),
        )
    )
    _print_startup(concierge, output_fn, resume)
    reader = _line_reader(input_fn)
    if sys.platform == "win32":
        output_fn(
            "On Windows, background progress updates while you type (console polling)."
            if reader.mode == "windows-tty"
            else "On Windows, background progress updates after you submit a line."
        )
    history = lineio.History()
    if reader.interactive:
        history.load_into_readline()
    interrupts = lineio.InterruptPolicy()
    prompt_pending = True
    try:
        while True:
            try:
                timeout_s = _console_poll_timeout(concierge, poll_timeout_s)
                line = reader.read("dip › " if prompt_pending else "", timeout_s)
            except EOFError:
                output_fn("")
                concierge.shutdown()
                return
            except KeyboardInterrupt:
                if _on_interrupt(concierge, interrupts, output_fn):
                    return
                prompt_pending = True
                continue
            if line is None:  # poll timeout — render background progress
                prompt_pending = False
                concierge.tick()
                continue
            prompt_pending = True
            interrupts.reset()
            history.append(line)
            if _dispatch_line(concierge, line, output_fn):
                return
            concierge.tick()
    finally:
        history.save()
