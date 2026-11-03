#!/usr/bin/env python3
"""Run the simulated paper order loop until the count or a stop signal.

Usage: paper_flow.py STATE COUNT PREFIX
Stdout is empty. The process exits 0 after a clean stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from quant_fund.observe.write_halt import DurableWriteError
from quant_fund.paper.durable_book import serve
from quant_fund.paper.trading_session import SessionCommandError


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        sys.stderr.write("usage: paper_flow.py STATE COUNT PREFIX\n")
        return 1
    prefix = argv[3]
    if not prefix.isascii() or not prefix.replace("-", "").isalnum():
        sys.stderr.write("prefix\n")
        return 1
    try:
        count = int(argv[2])
    except ValueError:
        sys.stderr.write("count\n")
        return 1
    if count < 0:
        sys.stderr.write("count\n")
        return 1
    try:
        return serve(Path(argv[1]), count, prefix)
    except (SessionCommandError, DurableWriteError, OSError):
        sys.stderr.write("paper_flow_failed\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
