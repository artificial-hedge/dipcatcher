"""``dipcatcher exec`` — one prompt, no interactive prompts, documented exit codes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_EXEC_EPILOG = """
exit codes:
  0    ok
  1    runtime, model, or tool error
  2    usage or configuration error
  3    answer withheld by the honesty gate
  4    time or usage budget exceeded (json status timeout or budget)
  5    no model endpoint configured for the selected model
  130  interrupted (SIGINT or SIGHUP)
  143  terminated (SIGTERM)

json status: ok, error, withheld, cancelled, timeout, budget, no_endpoint
"""

_MODELS = ("fx1", "fx1-lite")


def main(argv: Sequence[str] | None = None) -> None:
    """Run one exec turn. Raises ``SystemExit`` with the documented code."""
    parser = _parser()
    options = parser.parse_args(list(argv) if argv is not None else None)
    from fx1.interactive.headless import (
        EXIT_INTERRUPTED,
        EXIT_TERMINATED,
        EXIT_USAGE,
        ExecInterrupted,
        ExecRequest,
        ExecUsageError,
        run_exec,
    )

    try:
        prompt = _prompt_from(
            options.prompt,
            isatty=sys.stdin.isatty(),
            read=sys.stdin.read,
        )
        request = ExecRequest(
            prompt=prompt,
            model=_model(options.model),
            timeout_s=options.timeout_s,
            approve_keys=_approve_keys(options.approve or []),
            session=options.session,
            output_path=options.output_last_message,
            datasets=tuple(options.data or ()),
        )
        outcome = run_exec(request)
    except ExecUsageError as exc:
        _usage(str(exc), json_mode=options.json, model=_model_or_raw(options.model))
        raise SystemExit(EXIT_USAGE) from exc
    except ExecInterrupted as exc:
        code = EXIT_TERMINATED if exc.signum == _sigterm() else EXIT_INTERRUPTED
        print(f"interrupted by signal {exc.signum}", file=sys.stderr)
        raise SystemExit(code) from exc
    _emit(outcome, json_mode=options.json)
    raise SystemExit(outcome.exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dipcatcher exec",
        description=(
            "Send one line through the same path as chat. "
            "Plain mode prints only the final answer on stdout; "
            "progress and errors go to stderr. --json prints one document."
        ),
        epilog=_EXEC_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="fx1", help="fx1 or fx1-lite (default: fx1)")
    parser.add_argument("--json", action="store_true", help="print one dipcatcher.exec.v1 document")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="overall deadline in seconds, including background jobs (default: 300)",
    )
    parser.add_argument(
        "--approve",
        action="append",
        default=None,
        help="comma-separated approval keys such as harness:research (repeatable)",
    )
    parser.add_argument(
        "--session",
        default="none",
        help="none (default, ephemeral), new, last, or a session id",
    )
    parser.add_argument(
        "--output-last-message",
        default=None,
        help="write the final answer atomically with mode 0600",
    )
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="CSV or Parquet to attach for this turn (repeatable)",
    )
    parser.add_argument("prompt", nargs="*", help="prompt text, or '-' to read stdin")
    return parser


def _prompt_from(
    args: Sequence[str],
    *,
    isatty: bool,
    read: Callable[[], str],
) -> str:
    from fx1.interactive.headless import ExecUsageError

    if not args:
        if isatty:
            raise ExecUsageError("exec needs a prompt; stdin is a terminal so it will not block")
        return read()
    if list(args) == ["-"]:
        if isatty:
            raise ExecUsageError(
                "exec will not read a prompt from a terminal; pass the text as arguments"
            )
        return read()
    if "-" in args:
        raise ExecUsageError("pass the prompt as arguments or as '-', not both")
    return " ".join(args)


def _approve_keys(values: Sequence[str]) -> tuple[str, ...]:
    from fx1.interactive.headless import ExecUsageError

    keys: list[str] = []
    for value in values:
        for part in value.split(","):
            key = part.strip()
            if not key:
                raise ExecUsageError("empty approval key")
            keys.append(key)
    return tuple(keys)


def _model(value: str) -> str:
    from fx1.interactive.headless import ExecUsageError

    if value not in _MODELS:
        raise ExecUsageError(f"unknown model {value!r}; choose from {list(_MODELS)}")
    return value


def _model_or_raw(value: str) -> str:
    if value in _MODELS:
        return value
    return "fx1"


def _usage(message: str, *, json_mode: bool, model: str) -> None:
    from fx1.interactive.headless import EXEC_SCHEMA
    from fx1.interactive.redact import redact_text

    text = redact_text(message)
    print(text, file=sys.stderr)
    if not json_mode:
        return
    payload = {
        "schema": EXEC_SCHEMA,
        "status": "error",
        "model": model,
        "served_model": None,
        "answer": "",
        "usage": None,
        "latency_s": None,
        "evidence": None,
        "session_id": None,
        "warnings": [],
        "errors": [text],
        "duration_s": 0.0,
    }
    _write_json(payload)


def _emit(outcome: Any, *, json_mode: bool) -> None:
    if json_mode:
        _write_json(outcome.as_json())
    elif outcome.answer:
        text = outcome.answer
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
    for line in (*outcome.warnings, *outcome.errors):
        print(line, file=sys.stderr, flush=True)


def _write_json(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _sigterm() -> int:
    import signal

    sig = getattr(signal, "SIGTERM", None)
    if isinstance(sig, int):
        return sig
    return 15
