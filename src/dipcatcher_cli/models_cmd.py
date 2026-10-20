"""``dipcatcher models`` — presence list and offline or one-call checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> None:
    """List or check models. Raises ``SystemExit`` with 0, 1, 2, or 5."""
    parser = _parser()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"list", "check", "-h", "--help"}:
        args = ["list", *args]
    options = parser.parse_args(args)
    if options.command == "check":
        _check(options)
        return
    _list(options)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dipcatcher models",
        description="Show endpoint presence for fx1 and fx1-lite, or check them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listed = sub.add_parser("list", help="presence-only table of both models")
    listed.add_argument(
        "--json", action="store_true", help="print one dipcatcher.models.v1 document"
    )
    check = sub.add_parser("check", help="validate configuration, or ping with --live")
    check.add_argument("model", nargs="?", help="fx1 or fx1-lite; default is both")
    check.add_argument(
        "--live", action="store_true", help="place one identity call per configured model"
    )
    check.add_argument(
        "--timeout-s", type=float, default=30.0, help="per-model call budget in seconds"
    )
    check.add_argument(
        "--json", action="store_true", help="print one dipcatcher.models_check.v1 document"
    )
    check.add_argument(
        "--out", default=None, help="write the JSON record atomically with mode 0600"
    )
    return parser


def _list(options: argparse.Namespace) -> None:
    from fx1.interactive.models_check import format_model_list, list_payload, presence_report

    rows, warnings = presence_report()
    if options.json:
        _write_json(list_payload(rows, warnings))
        return
    sys.stdout.write(format_model_list(rows, warnings) + "\n")


def _check(options: argparse.Namespace) -> None:
    from fx1.interactive.atomicstore import atomic_write_json
    from fx1.interactive.models_check import check_models, format_check_report, report_exit_code
    from fx1.interactive.redact import redact_text

    try:
        report = check_models(
            _models(options.model), live=options.live, timeout_s=options.timeout_s
        )
    except (KeyError, ValueError) as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        raise SystemExit(2) from exc
    payload = report.as_json()
    if options.out:
        atomic_write_json(Path(options.out), payload, mode=0o600)
    if options.json:
        _write_json(payload)
    else:
        sys.stdout.write(format_check_report(report) + "\n")
    sys.stdout.flush()
    raise SystemExit(report_exit_code(report))


def _models(model: str | None) -> tuple[str, ...] | None:
    if not model:
        return None
    return (model,)


def _write_json(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
