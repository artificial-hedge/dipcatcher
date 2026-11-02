"""Fail unless ``.tmp_security/summary.json`` is a passing scan of HEAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from security_gate_lib import evaluate_summary, git_head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    summary_path = root / ".tmp_security" / "summary.json"
    if not summary_path.is_file():
        print("security_gate summary missing")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = evaluate_summary(summary, git_head(root))
    if errors:
        print("security_gate " + ",".join(errors))
        return 1
    print("security_gate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
