"""Generate discovery records, extension wrappers, and declaration shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from fx1.capabilities import (
    EXTENSIONS_PATH,
    SEED_COUNT,
    SEED_PATH,
    write_declaration_shards,
    write_extension_modules,
    write_seed_catalog,
)


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
        help="Number of deterministic capability registrations to write.",
    )
    parser.add_argument(
        "--extensions-out",
        type=Path,
        default=EXTENSIONS_PATH,
        help="Output package root for generated extension wrappers.",
    )
    parser.add_argument(
        "--declarations-out",
        type=Path,
        default=Path(__file__).with_name("generated_capability_declarations"),
        help="Output root for per-extension executable registration source.",
    )
    args = parser.parse_args()
    count = write_seed_catalog(args.out, count=args.count)
    modules = write_extension_modules(args.extensions_out)
    source_count = write_declaration_shards(args.declarations_out, count=args.count)
    print(f"seeded {count:,} generated capability recipes at {args.out}")
    print(f"wrote {modules} generated extension wrappers at {args.extensions_out}")
    print(f"wrote {source_count:,} executable registration lines at {args.declarations_out}")


if __name__ == "__main__":
    main()
