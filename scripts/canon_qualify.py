#!/usr/bin/env python
"""Emit the deterministic corpus qualification audit (full dump + summary).

Runs :func:`quant_fund.models.canon_qualification.audit_tree` over the repo and
writes two canonical JSON artifacts (two runs over the same tree are
byte-identical):

- ``.dsh-24x7/canon_qualification_audit.json`` — the complete audit dump
  (per-file verdicts and evidence, family/wiring maps, import-graph
  quarantine plan with component ids and blocking importers). Derived and
  gitignored: tens of MB, never checked in.
- ``quality/canon_qualification_summary.json`` — the small checked-in
  summary (kept under ~300 KB so it stays reviewable in git): counts per
  verdict, the retired and live-candidate family names, ruleset version +
  hash, a ``models_tree_sha256`` digest binding the audited bytes of
  ``src/quant_fund/models/*.py`` to this summary (the fast regression guard
  recomputes it from the live tree), and the full dump's path + SHA-256.

Usage::

    uv run python scripts/canon_qualify.py                          # regenerate
    uv run python scripts/canon_qualify.py --check                  # fail if stale
    uv run python scripts/canon_qualify.py --emit-retired-families \
        src/quant_fund/research/catalog/retired_families.py         # codegen

``--emit-retired-families`` generates the registry retirement mapping from the
on-disk summary (never hand-written), so the audit and the registry cannot
drift apart silently. The audit is evidence for corpus-integrity claims; it
never modifies the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_fund.models import canon_qualification as cq  # noqa: E402

DEFAULT_DUMP = Path(".dsh-24x7/canon_qualification_audit.json")
DEFAULT_SUMMARY = Path("quality/canon_qualification_summary.json")

#: RESEARCH_RECEIPT_SCHEMA_VERSION at the 2026-10-07 retirement: receipts up
#: to and including this schema may reference the retired families.
LAST_RECEIPT_SCHEMA_VERSION = 2

REASON_TEMPLATE_STUB = (
    "template-shaped constant-score family, no behavioral mechanism: every wired "
    "module is a NON_QUALIFYING_TEMPLATE stub (shared AST shape, constant-check "
    "bench over constant arguments, zero control flow, zero data parameters); "
    "retired per docs/BENCHMARK_FAMILY_LIFECYCLE.md; evidence: "
    "quality/canon_qualification_summary.json"
)

REASON_PARTIALLY_TEMPLATE = (
    "template-shaped constant-score family, partially template-backed: its wired "
    "modules include NON_QUALIFYING_TEMPLATE stubs (shared AST shape, "
    "constant-check bench over constant arguments, zero control flow, zero data "
    "parameters); retired per docs/BENCHMARK_FAMILY_LIFECYCLE.md; evidence: "
    "quality/canon_qualification_summary.json"
)


def _summary_text(audit: dict[str, Any]) -> str:
    """Human-readable console echo of the headline counts."""
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


def _family_names(audit: dict[str, Any], classification: str) -> list[str]:
    families = audit["families"]
    assert isinstance(families, list)
    return sorted(row["family_id"] for row in families if row["classification"] == classification)


def _models_tree_sha256(files: list[dict[str, Any]]) -> str:
    """Digest binding the audited bytes of live ``src/quant_fund/models/*.py``."""
    rows = sorted(
        (row["path"], row["source_sha256"])
        for row in files
        if row["location"] == "live" and row["path"].startswith("src/quant_fund/models/")
    )
    payload = "".join(f"{path} {sha}\n" for path, sha in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_summary(audit: dict[str, Any], dump_text: str, dump_rel: str) -> dict[str, Any]:
    """Derive the small checked-in summary from a full audit dump."""
    files = audit["files"]
    families = audit["families"]
    assert isinstance(files, list) and isinstance(families, list)
    retired = _family_names(audit, "RETIRED_CANDIDATE")
    partial = sorted(
        row["family_id"]
        for row in families
        if row["classification"] == "RETIRED_CANDIDATE" and "QUALIFYING" in row["module_verdicts"]
    )
    dump_bytes = dump_text.encode("utf-8")
    return {
        "schema": "canon_qualification_summary.v2",
        "ruleset_version": cq.RULESET_VERSION,
        "ruleset_hash": audit["ruleset_hash"],
        "counts": audit["counts"],
        "retired_families": retired,
        "retired_families_partially_template": partial,
        "live_candidate_families": _family_names(audit, "LIVE_CANDIDATE"),
        "models_tree_sha256": _models_tree_sha256(files),
        "full_dump": {
            "path": dump_rel,
            "sha256": hashlib.sha256(dump_bytes).hexdigest(),
            "bytes": len(dump_bytes),
        },
    }


def _generated_module_text(summary: dict[str, Any]) -> str:
    """Render ``catalog/retired_families.py`` from the summary."""
    retired = summary["retired_families"]
    partial = set(summary["retired_families_partially_template"])
    stub_names = sorted(set(retired) - partial)
    partial_names = sorted(partial)
    dump = summary["full_dump"]
    header = f'''"""GENERATED retired optional benchmark families — DO NOT HAND-EDIT.

Regenerate with::

    uv run python scripts/canon_qualify.py
    uv run python scripts/canon_qualify.py --emit-retired-families \\
        src/quant_fund/research/catalog/retired_families.py

Derived from ``quality/canon_qualification_summary.json``
(ruleset_hash={summary["ruleset_hash"]}; full dump ``{dump["path"]}``
sha256={dump["sha256"]}). OPTIONAL -> RETIRED per
``docs/BENCHMARK_FAMILY_LIFECYCLE.md``. ``OPTIONAL_BENCHMARK_FAMILIES`` stays
the append-only accepted set, so archived receipts naming these families keep
verifying; each entry records its retirement reason and the last receipt
schema version that could have carried the family.
"""

from typing import TypedDict


class RetiredFamilyRecord(TypedDict):
    """Retirement evidence for one optional benchmark family."""

    reason: str
    last_receipt_schema_version: int


REASON_TEMPLATE_STUB = (
    "template-shaped constant-score family, no behavioral mechanism: every wired "
    "module is a NON_QUALIFYING_TEMPLATE stub (shared AST shape, constant-check "
    "bench over constant arguments, zero control flow, zero data parameters); "
    "retired per docs/BENCHMARK_FAMILY_LIFECYCLE.md; evidence: "
    "quality/canon_qualification_summary.json"
)

REASON_PARTIALLY_TEMPLATE = (
    "template-shaped constant-score family, partially template-backed: its wired "
    "modules include NON_QUALIFYING_TEMPLATE stubs (shared AST shape, "
    "constant-check bench over constant arguments, zero control flow, zero data "
    "parameters); retired per docs/BENCHMARK_FAMILY_LIFECYCLE.md; evidence: "
    "quality/canon_qualification_summary.json"
)

#: RESEARCH_RECEIPT_SCHEMA_VERSION at the 2026-10-07 retirement.
LAST_RECEIPT_SCHEMA_VERSION = {LAST_RECEIPT_SCHEMA_VERSION}
'''

    def _tuple_block(var: str, names: list[str], *, leading_blank: bool) -> list[str]:
        # ruff-clean emission: an empty tuple prints as `()` on one line, so
        # generated code passes `ruff format --check` without a post-hoc pass.
        head = f"\n{var}" if leading_blank else var
        if not names:
            return [f"{head}: tuple[str, ...] = ()", ""]
        return [f"{head}: tuple[str, ...] = ("] + [f'    "{name}",' for name in names] + [")", ""]

    body = _tuple_block("_TEMPLATE_STUB_FAMILIES", stub_names, leading_blank=True)
    body += _tuple_block("_PARTIALLY_TEMPLATE_FAMILIES", partial_names, leading_blank=False)
    body.append("RETIRED_BENCHMARK_FAMILIES: dict[str, RetiredFamilyRecord] = {}")
    body.append("for _name in _TEMPLATE_STUB_FAMILIES:")
    body.append("    RETIRED_BENCHMARK_FAMILIES[_name] = RetiredFamilyRecord(")
    body.append("        reason=REASON_TEMPLATE_STUB,")
    body.append("        last_receipt_schema_version=LAST_RECEIPT_SCHEMA_VERSION,")
    body.append("    )")
    body.append("for _name in _PARTIALLY_TEMPLATE_FAMILIES:")
    body.append("    RETIRED_BENCHMARK_FAMILIES[_name] = RetiredFamilyRecord(")
    body.append("        reason=REASON_PARTIALLY_TEMPLATE,")
    body.append("        last_receipt_schema_version=LAST_RECEIPT_SCHEMA_VERSION,")
    body.append("    )")
    body.append("")
    return header + "\n".join(body)


def _emit_retired_families(summary_path: Path, out_path: Path) -> int:
    """Codegen the retirement mapping from the on-disk summary."""
    if not summary_path.is_file():
        print(f"missing summary: {summary_path}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_generated_module_text(summary), encoding="utf-8")
    print(f"wrote {out_path} ({len(summary['retired_families'])} retired families)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Write (or verify) the audit artifacts; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--dump", type=Path, default=None, help="full dump path")
    parser.add_argument("--summary", type=Path, default=None, help="summary path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail (exit 1) if either artifact differs from a fresh run",
    )
    parser.add_argument(
        "--emit-retired-families",
        type=Path,
        default=None,
        metavar="PATH",
        help="generate catalog/retired_families.py from the summary and exit",
    )
    args = parser.parse_args(argv)
    root: Path = args.root
    if args.emit_retired_families is not None:
        summary_path = args.summary if args.summary is not None else root / DEFAULT_SUMMARY
        return _emit_retired_families(summary_path, args.emit_retired_families)

    dump_path: Path = args.dump if args.dump is not None else root / DEFAULT_DUMP
    summary_path = args.summary if args.summary is not None else root / DEFAULT_SUMMARY
    dump_rel = (
        dump_path.relative_to(root).as_posix() if dump_path.is_relative_to(root) else str(dump_path)
    )

    audit = cq.audit_tree(root)
    dump_text = cq.dumps_audit(audit)
    summary = _build_summary(audit, dump_text, dump_rel)
    summary_text = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True) + "\n"

    if args.check:
        stale = [
            path
            for path, text in ((dump_path, dump_text), (summary_path, summary_text))
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"STALE: {', '.join(str(p) for p in stale)}", file=sys.stderr)
            return 1
        print(f"OK: {dump_path} and {summary_path} match a fresh audit run")
        return 0

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(dump_text, encoding="utf-8")
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"wrote {dump_path} ({len(dump_text)} bytes)")
    print(f"wrote {summary_path} ({len(summary_text)} bytes)")
    print(_summary_text(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
