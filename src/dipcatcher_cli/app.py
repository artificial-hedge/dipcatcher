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


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _version_line() -> str:
    import fx1

    return f"dipcatcher {fx1.__version__} (fx-1 harness; models fx1, fx1-lite)"


def _split_leading_model(args: Sequence[str]) -> tuple[str | None, list[str]]:
    """Peel a leading ``--model/--profile/-p`` flag used by both chat and fxi."""
    if len(args) >= 2 and args[0] in _MODEL_FLAGS:
        return args[1], list(args[2:])
    return None, list(args)


def _chat(args: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="dipcatcher chat",
        description="Chat with fx1 or fx1-lite; /help lists research and session controls.",
        epilog=(
            "In chat: /superpower <goal>, /research, /status, /keys, /doctor, /exit.\n"
            "Session journal: $FX1_CONFIG_DIR/sessions/*.jsonl · "
            "Ctrl-C once cancels the running job, twice exits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=_CHAT_MODELS, default=_DEFAULT_CHAT_MODEL)
    parser.add_argument(
        "--resume",
        metavar="SESSION",
        default=None,
        help="show a prior session's transcript (id or unique prefix) before chatting",
    )
    options = parser.parse_args(args)
    if not _interactive():
        parser.error(
            "chat requires an interactive terminal on stdin and stdout; "
            "use explicit lab commands or one-shot commands "
            "(dipcatcher superpower, dipcatcher sessions) for scripts"
        )

    from fx1.interactive.concierge import run_console

    run_console(model=options.model, resume=options.resume)


def _print_launcher_help() -> None:
    """Short chat-first banner before the lab Typer help tree."""
    print("Start a conversation: dipcatcher (interactive terminal)")
    print("Choose a model: dipcatcher chat --model fx1-lite")
    print("Resume a session: dipcatcher chat --resume <id>")
    print('One-shot research: dipcatcher superpower "<goal>" --sync')
    print("Sessions: dipcatcher sessions   |  dipcatcher resume <id>")
    print("Endpoints: dipcatcher keys list  |  dipcatcher keys set fx1")
    print("In chat: /help, /superpower, /status, /use <model>, /doctor")
    print("Version: dipcatcher --version")
    print("Lab / scripts: dipcatcher <command> --help\n")


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


def main(argv: Sequence[str] | None = None) -> None:
    """Open chat in a terminal; retain the existing lab command/exit contracts.

    Lab/help paths propagate Typer's ``SystemExit`` so scripts and tests see
    the same exit codes as calling ``quant_fund.cli.main.app`` directly.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 1 and args[0] in VERSION_FLAGS:
        print(_version_line())
        return

    model, rest = _split_leading_model(args)
    if rest[:1] == [CHAT_COMMAND]:
        chat_args = list(rest[1:])
        if model is not None:
            chat_args = ["--model", model, *chat_args]
        _chat(chat_args)
        return
    if not rest and _interactive():
        _chat([] if model is None else ["--model", model])
        return
    if rest and rest[0] in ONESHOT_COMMANDS:
        _oneshot(args)
        return
    if args and args[0].startswith("/") and not args[0].startswith("//"):
        print(_slash_argv_message(args[0]), file=sys.stderr)
        raise SystemExit(2)

    # Import the lab only on the automation/help path. No library imports this
    # outer launcher; quant_fund continues to have no dependency on the UI.
    from quant_fund.cli.main import app

    if not args or args in (["--help"], ["-h"]):
        _print_launcher_help()
    app(args=args or ["--help"], prog_name="dipcatcher")
