"""Run the reality filter gate on POSIX and Windows with the same sequence.

The default provenance DB is gitignored. When it is absent, a nonempty
committed trial ledger takes precedence; otherwise preflight emits an explicit
skip. Scoring commands keep their own fail-closed exit codes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path("data/metadata/proofcore.duckdb")
DEFAULT_LEDGER = Path("data/metadata/proofcore-trials.jsonl")
DEFAULT_COMMITTED_LEDGER = Path("research/reality/trials.jsonl")
SKIP_EXIT = 3


def _quant(*args: str) -> int:
    # Use the console script installed beside this interpreter so the gate
    # cannot silently dispatch to a different ``quant`` on PATH.
    launcher = Path(sys.executable).with_name("quant.exe" if sys.platform == "win32" else "quant")
    try:
        return subprocess.run(
            [str(launcher), *args],
            cwd=ROOT,
            check=False,
        ).returncode
    except OSError as exc:
        print(f"REALITY_FILTER_ERROR: cannot launch quant: {exc}", file=sys.stderr)
        return 2


def _score(ledger: Path) -> int:
    report_code = _quant("reality", "trial-report", "--ledger", str(ledger))
    if report_code != 0:
        return report_code
    return _quant("reality", "ledger-gate", "--ledger", str(ledger))


def run_gate(db: Path, ledger: Path, committed_ledger: Path) -> int:
    """Return 0 for a pass or explicit skip; propagate any scoring failure."""
    if (
        db == DEFAULT_DB
        and not (ROOT / db).is_file()
        and (ROOT / committed_ledger).is_file()
        and (ROOT / committed_ledger).stat().st_size > 0
    ):
        code = _quant("reality", "preflight", "--ledger", str(committed_ledger))
        if code == SKIP_EXIT:
            return 0
        if code != 0:
            return code
        return _score(committed_ledger)

    code = _quant("reality", "preflight", "--db", str(db))
    if code == SKIP_EXIT:
        return 0
    if code != 0:
        return code
    code = _quant("proofcore", "export", "--db", str(db), "--out", str(ledger))
    if code != 0:
        return code
    code = _quant("reality", "preflight", "--ledger", str(ledger))
    if code == SKIP_EXIT:
        return 0
    if code != 0:
        return code
    return _score(ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--committed-ledger", type=Path, default=DEFAULT_COMMITTED_LEDGER)
    args = parser.parse_args()
    return run_gate(args.db, args.ledger, args.committed_ledger)


if __name__ == "__main__":
    raise SystemExit(main())
