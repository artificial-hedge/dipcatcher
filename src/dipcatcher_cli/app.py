"""Route interactive dipcatcher sessions without coupling the research engine to fx1.

Explicit lab commands pass through unchanged, including calls made by the
model's registered harness subprocess runner. Bare redirected invocation shows
help without reading input, credentials, or starting a conversation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

CHAT_COMMAND = "chat"
# Commands this launcher handles itself; everything else goes to the lab app.
LAUNCHER_COMMANDS: frozenset[str] = frozenset({CHAT_COMMAND})


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _chat(args: Sequence[str]) -> None:
    from fx1.interactive.profiles import DEFAULT_MODEL, MODELS

    parser = argparse.ArgumentParser(
        prog="dipcatcher chat",
        description="Chat with fx1 or fx1-lite; /help lists research and session controls.",
    )
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    options = parser.parse_args(args)
    if not _interactive():
        parser.error(
            "chat requires an interactive terminal on stdin and stdout; "
            "use explicit lab commands or fxi one-shot commands for scripts"
        )

    from fx1.interactive.concierge import run_console

    run_console(model=options.model)


def main(argv: Sequence[str] | None = None) -> None:
    """Open chat in a terminal; retain the existing lab command/exit contracts."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == [CHAT_COMMAND]:
        _chat(args[1:])
        return
    if not args and _interactive():
        _chat([])
        return

    # Import the lab only on the automation/help path. No library imports this
    # outer launcher; quant_fund continues to have no dependency on the UI.
    from quant_fund.cli.main import app

    if not args or args in (["--help"], ["-h"]):
        print("Start a conversation: dipcatcher (interactive terminal)")
        print("Choose a model: dipcatcher chat --model fx1-lite\n")
    app(args=args or ["--help"], prog_name="dipcatcher")
