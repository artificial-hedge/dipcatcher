#!/usr/bin/env python3
"""Backup or restore the paper ledger and checkpoints.

The archive stores hashes. Restore verifies every hash before it writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_fund.paper.state_backup import PaperBackupError, backup_paper_state, restore_paper_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backup or restore paper state")
    sub = parser.add_subparsers(dest="command", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--source", type=Path, required=True)
    backup.add_argument("--dest", type=Path, required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            backup_paper_state(args.source, args.dest)
        else:
            restore_paper_state(args.archive, args.dest)
    except PaperBackupError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
