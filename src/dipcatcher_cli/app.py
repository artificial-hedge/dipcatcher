"""Route interactive dipcatcher sessions without coupling the research engine to fx1.

Explicit lab commands pass through unchanged, including calls made by the
model's registered harness subprocess runner. Bare redirected invocation shows
help without reading input, credentials, or starting a conversation.

Concierge one-shots (``superpower``, ``sessions``, ``resume``, ``keys``,
``flash``, ``setup``) are owned here so the product name matches the docs:
``dipcatcher superpower …`` is the same path as ``fxi superpower …``. Names
in this set must not collide with lab Typer commands (``research``,
``doctor``, ``eval``, …).
Slash commands belong in the chat; a stray ``/superpower`` on the shell is
explained rather than forwarded to the lab as an unknown subcommand.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

CHAT_COMMAND = "chat"
VERSION_FLAGS: frozenset[str] = frozenset({"--version", "-V"})
_MODEL_FLAGS: frozenset[str] = frozenset({"--model", "--profile", "-p"})
# One-shot orchestration lives in ``fx1.interactive.app`` (also ``fxi``).
ONESHOT_COMMANDS: frozenset[str] = frozenset(
    {"superpower", "sessions", "resume", "keys", "flash", "setup"}
)
# Commands this launcher handles itself; everything else goes to the lab app.
LAUNCHER_COMMANDS: frozenset[str] = frozenset({CHAT_COMMAND}) | ONESHOT_COMMANDS
# Keep chat argparse free of fx1.interactive imports (profiles pulls serve URLs).
_CHAT_MODELS: tuple[str, ...] = ("fx1", "fx1-lite")
_DEFAULT_CHAT_MODEL = "fx1"

_CHILD_OFF = frozenset({"", "0", "false", "no", "off"})


def _harness_child() -> bool:
    """True when spawned by the concierge harness runner (must not open chat)."""
    raw = os.environ.get("DIPCATCHER_CHILD")
    if raw is None:
        return False
    return raw.strip().lower() not in _CHILD_OFF


def _interactive() -> bool:
    if _harness_child():
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _version_line() -> str:
    import fx1

    return (
        f"dipcatcher {fx1.__version__} "
        "(fx-1 harness API; chat ids fx1, fx1-lite; no released weights)"
    )


def _split_leading_model(args: Sequence[str]) -> tuple[str | None, list[str]]:
    """Peel a leading ``--model/--profile/-p`` flag used by both chat and fxi."""
    if len(args) >= 2 and args[0] in _MODEL_FLAGS:
        return args[1], list(args[2:])
    return None, list(args)


def _chat(args: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="dipcatcher chat",
        description=(
            "Chat using the fx1 or fx1-lite chat id. No released weights. "
            "/help lists research and session controls."
        ),
        epilog=(
            "In chat: /superpower <goal>, /research, /status, /keys, /doctor, /exit.\n"
            "Session journal: $FX1_CONFIG_DIR/sessions/*.jsonl · "
            "Ctrl-C once cancels the running job, twice exits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=_CHAT_MODELS,
        default=_DEFAULT_CHAT_MODEL,
        help="model identifier; model turns require a configured endpoint",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION",
        help="resume the latest session, or a prior transcript by id or unique prefix",
    )
    options = parser.parse_args(args)
    if not _interactive():
        parser.error(
            "chat requires an interactive terminal on stdin and stdout; "
            "use explicit lab commands or one-shot commands "
            "(dipcatcher superpower, dipcatcher sessions) for scripts"
        )

    from fx1.interactive.console import run_console

    model_explicit = any(token in {"--model", "--profile", "-p"} for token in args)
    run_console(model=options.model, resume=options.resume, model_explicit=model_explicit)


def _print_launcher_help() -> None:
    """Short chat-first banner before the lab Typer help tree."""
    print("Start a conversation: dipcatcher (interactive terminal)")
    print("Choose a chat id: dipcatcher chat --model fx1-lite (no released weights)")
    print("Resume a session: dipcatcher chat --resume <id>")
    print('One-shot research: dipcatcher superpower "<goal>" --sync')
    print("Sessions: dipcatcher sessions   |  dipcatcher resume <id>")
    print("Endpoints: dipcatcher keys list  |  dipcatcher keys set fx1")
    print("In chat: /help, /superpower, /status, /use <model>, /doctor")
    print("Version: dipcatcher --version")
    print("Lab / scripts: dipcatcher <command> --help\n")


def _exec(args: Sequence[str]) -> None:
    """One prompt, no interactive read. ``exec_cmd`` raises ``SystemExit``."""
    from dipcatcher_cli.exec_cmd import main as exec_main

    exec_main(list(args))


def _sessions(args: Sequence[str]) -> None:
    """``sessions list|show|export|rm|prune``. ``sessions_cmd`` raises ``SystemExit``."""
    from dipcatcher_cli.sessions_cmd import sessions_main

    sessions_main(list(args))


def _resume(args: Sequence[str]) -> None:
    """Open chat on a saved session. ``resume_main`` raises ``SystemExit``."""
    from dipcatcher_cli.sessions_cmd import resume_main

    resume_main(list(args))


def _oneshot(args: Sequence[str]) -> None:
    """Run a concierge one-shot through the fxi Typer app under this prog name."""
    from fx1.interactive.app import app as fxi_app

    fxi_app(args=list(args), prog_name="dipcatcher")


def _slash_argv_message(command: str) -> str:
    name = command.split()[0]
    return (
        f"{name} is a chat slash command, not a lab subcommand.\n"
        "Open an interactive terminal and run `dipcatcher`, then type the slash command.\n"
        'Scripts: `dipcatcher superpower --sync "..."` or `dipcatcher research --help`.'
    )


def _forward_lab(args: Sequence[str]) -> None:
    from quant_fund.cli.main import app

    if not args or args in (["--help"], ["-h"]):
        _print_launcher_help()
    lab_args = ["--help"] if not args or list(args) == ["-h"] else list(args)
    app(args=lab_args, prog_name="dipcatcher")


def _with_model(args: Sequence[str], model: str | None) -> list[str]:
    """Prefix chat-style args with the leading model when one was given."""
    return list(args) if model is None else ["--model", model, *args]


def main(argv: Sequence[str] | None = None) -> None:
    """Open chat in a terminal; retain the existing lab command/exit contracts."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 1 and args[0] in VERSION_FLAGS:
        print(_version_line())
        return

    model, rest = _split_leading_model(args)
    if rest[:1] == [CHAT_COMMAND]:
        _chat(_with_model(rest[1:], model))
        return
    if not rest and _interactive():
        _chat([] if model is None else ["--model", model])
        return
    if rest[:1] == ["exec"]:
        _exec(rest[1:])
        return
    if rest[:1] == ["sessions"]:
        _sessions(rest[1:])
        return
    if rest[:1] == ["resume"]:
        _resume(_with_model(rest[1:], model))
        return
    if rest and rest[0] in ONESHOT_COMMANDS:
        _oneshot(args)
        return
    if args and args[0].startswith("/") and not args[0].startswith("//"):
        print(_slash_argv_message(args[0]), file=sys.stderr)
        raise SystemExit(2)
    _forward_lab(args)
