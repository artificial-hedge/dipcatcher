#!/usr/bin/env python
"""Emit the deterministic corpus qualification audit.

Runs :func:`quant_fund.models.canon_qualification.audit_tree` over the repo and
writes ``quality/canon_qualification_audit.json`` (canonical JSON: sorted keys,
stable separators, trailing newline) so two runs over the same tree are
byte-identical. The audit records per-file verdicts (QUALIFYING /
NON_QUALIFYING_TEMPLATE), family ids and waves, the pinned ``ruleset_hash``,
before/after (live vs attic) counts and the import-graph quarantine plan with
blocking importers for anything that cannot move safely.

Usage::

    uv run python scripts/canon_qualify.py            # regenerate the audit
    uv run python scripts/canon_qualify.py --check    # fail if it is stale

The audit is evidence for corpus-integrity claims; it never modifies the tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_fund.models import canon_qualification as cq  # noqa: E402


def _summary(audit: dict[str, object]) -> str:
    counts = audit["counts"]
    assert isinstance(counts, dict)
    lines = [f"ruleset_hash: {audit['ruleset_hash']}"]
    for section in ("baseline_corpus", "live_tree", "attic"):
        block = counts[section]
        assert isinstance(block, dict)
        lines.append(f"{section}: " + ", ".join(f"{k}={v}" for k, v in sorted(block.items())))
    outstanding = audit["outstanding_quarantine"]
    assert isinstance(outstanding, list)
    lines.append(f"outstanding_quarantine: {len(outstanding)} file(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write (or verify) the audit JSON; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (default: <root>/quality/canon_qualification_audit.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail (exit 1) if the on-disk audit differs from a fresh run",
    )
    args = parser.parse_args(argv)
    root: Path = args.root
    output: Path = (
        args.output if args.output is not None else root / "quality/canon_qualification_audit.json"
    )

    audit = cq.audit_tree(root)
    text = cq.dumps_audit(audit)

    if args.check:
        current = output.read_text(encoding="utf-8") if output.is_file() else None
        if current != text:
            print(f"STALE: {output} does not match a fresh audit run", file=sys.stderr)
            return 1
        print(f"OK: {output} matches a fresh audit run")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    print(_summary(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
