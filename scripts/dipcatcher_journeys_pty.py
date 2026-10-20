"""Run the frozen 100-journey suite and write a dipcatcher.journeys.v1 result.

Exit 0 only when at least 99 journeys pass and every critical journey passes.
A filtered run that covers fewer than 99 journeys exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen dipcatcher journey suite.")
    parser.add_argument("--out", required=True, help="Result JSON path")
    parser.add_argument(
        "--filter",
        default="",
        help="Comma-separated ids, categories, modes, or 'critical'",
    )
    args = parser.parse_args(argv)
    from tests.fx1.journeys_pty.runner import gate_satisfied, run_suite
    from tests.fx1.journeys_pty.textutil import redact

    payload = run_suite(args.filter)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    text = redact(text)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return 0 if gate_satisfied(payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())
