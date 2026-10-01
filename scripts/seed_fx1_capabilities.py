"""Generate the compact, reproducible one-million-record fx-1 capability seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from fx1.capabilities import SEED_COUNT, SEED_PATH, write_seed_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=SEED_PATH,
        help="Output .bin.gz path (defaults to the bundled fx1 package catalog).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=SEED_COUNT,
        help="Number of deterministic catalog records to write.",
    )
    args = parser.parse_args()
    count = write_seed_catalog(args.out, count=args.count)
    print(f"seeded {count:,} generated capability recipes at {args.out}")


if __name__ == "__main__":
    main()
