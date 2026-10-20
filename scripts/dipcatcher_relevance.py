#!/usr/bin/env python3
"""Score the frozen dipcatcher goal/data relevance suite.

Writes one ``dipcatcher.relevance.v1`` JSON document. Exits 1 when a plan
invariant fails, a model-refinement attack leaves the typed catalog, or an
unsafe case is not refused. A supported-case miss is recorded and does not
by itself change the exit code. The expert half of gate P3 stays open.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.fx1.relevance.runner import CASES_PATH, run_suite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the frozen relevance suite.")
    parser.add_argument("--out", required=True, type=Path, help="JSON output path.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="Frozen cases file.")
    args = parser.parse_args(argv)
    result = _run(args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


def _run(cases: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="dipcatcher-relevance-") as tmp:
        root = Path(tmp)
        _set_env(root)
        return run_suite(work_dir=root / "work", cases_path=cases)


def _set_env(root: Path) -> None:
    os.environ["FX1_CONFIG_DIR"] = str(root / "fx1-config")
    os.environ["FX1_SESSION_DIR"] = str(root / "fx1-sessions")
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"
    os.environ["FORCE_COLOR"] = "0"


if __name__ == "__main__":
    raise SystemExit(main())
