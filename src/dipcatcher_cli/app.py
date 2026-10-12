"""Route interactive dipcatcher sessions without coupling the research engine to fx1.

Explicit lab commands pass through unchanged, including calls made by the
model's registered harness subprocess runner. Bare redirected invocation shows
help without reading input, credentials, or starting a conversation.
``exec``, ``models``, ``version``, ``sessions``, and ``resume`` are launcher
commands and are imported only when those commands run, so lab passthrough
and ``--help`` do not import the chat stack.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

CHAT_COMMAND = "chat"
EXEC_COMMAND = "exec"
MODELS_COMMAND = "models"
VERSION_COMMAND = "version"
SESSIONS_COMMAND = "sessions"
RESUME_COMMAND = "resume"
# Commands this launcher handles itself; everything else goes to the lab app.
LAUNCHER_COMMANDS: frozenset[str] = frozenset(
    {
        CHAT_COMMAND,
        EXEC_COMMAND,
        MODELS_COMMAND,
        VERSION_COMMAND,
        SESSIONS_COMMAND,
        RESUME_COMMAND,
    }
)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _chat(args: Sequence[str]) -> None:
    from fx1.interactive.profiles import DEFAULT_MODEL, MODELS

    parser = argparse.ArgumentParser(
        prog="dipcatcher chat",
        description="Chat with fx1 or fx1-lite; /help lists research and session controls.",
    )
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reload the latest saved session. An interrupted job is not restarted.",
    )
    options = parser.parse_args(args)
    if not _interactive():
        parser.error(
            "chat requires an interactive terminal on stdin and stdout; "
            "use explicit lab commands or fxi one-shot commands for scripts"
        )

    from fx1.interactive.concierge import run_console

    model = options.model or DEFAULT_MODEL
    if options.resume:
        run_console(model=model, resume=True, model_explicit=options.model is not None)
    else:
        run_console(model=model)


def _exec(args: Sequence[str]) -> None:
    from dipcatcher_cli.exec_cmd import main as exec_main

    exec_main(args)


def _models(args: Sequence[str]) -> None:
    from dipcatcher_cli.models_cmd import main as models_main

    models_main(args)


def _sessions(args: Sequence[str]) -> None:
    from dipcatcher_cli.sessions_cmd import sessions_main

    sessions_main(args)


def _resume(args: Sequence[str]) -> None:
    from dipcatcher_cli.sessions_cmd import resume_main

    resume_main(args)


def _version() -> None:
    from fx1 import __version__

    print(f"dipcatcher {__version__}")


def _print_launcher_preface() -> None:
    print("Start a conversation: dipcatcher (interactive terminal)")
    print("Choose a model: dipcatcher chat --model fx1-lite")
    print('Script one turn: dipcatcher exec "..."  (exit codes: dipcatcher exec --help)')
    print("Check models: dipcatcher models list | dipcatcher models check --live")
    print()


def _dispatch_launcher(args: Sequence[str]) -> bool:
    head = args[:1]
    if head == [CHAT_COMMAND]:
        _chat(args[1:])
        return True
    if head == [EXEC_COMMAND]:
        _exec(args[1:])
        return True
    if head == [MODELS_COMMAND]:
        _models(args[1:])
        return True
    if head == [SESSIONS_COMMAND]:
        _sessions(args[1:])
        return True
    if head == [RESUME_COMMAND]:
        _resume(args[1:])
        return True
    if args == ["--version"] or head == [VERSION_COMMAND]:
        _version_args(args)
        return True
    if not args and _interactive():
        _chat([])
        return True
    return False


def _version_args(args: Sequence[str]) -> None:
    if args == ["--version"] or args == [VERSION_COMMAND]:
        _version()
        return
    print("usage: dipcatcher version", file=sys.stderr)
    raise SystemExit(2)


def _forward_lab(args: Sequence[str]) -> None:
    # Import the lab only on the automation/help path. No library imports this
    # outer launcher; quant_fund continues to have no dependency on the UI.
    from quant_fund.cli.main import app

    if not args or args in (["--help"], ["-h"]):
        _print_launcher_preface()
    app(args=args or ["--help"], prog_name="dipcatcher")


def main(argv: Sequence[str] | None = None) -> None:
    """Open chat in a terminal; retain the existing lab command/exit contracts."""
    args = list(sys.argv[1:] if argv is None else argv)
    if _dispatch_launcher(args):
        return
    _forward_lab(args)
