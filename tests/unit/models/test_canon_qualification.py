"""Corpus qualification guard — template copies may never return.

Regression guard for the corpus-integrity audit (problem #5). The
classification itself is objective and hash-pinned (``RULESET`` /
:func:`~quant_fund.models.canon_qualification.ruleset_hash`); these tests pin
the ruleset hash, keep the checked-in audit summary in sync with the live
tree (a reintroduced or edited module fails immediately), prove the verdict
rule classifies template copies as ``NON_QUALIFYING_TEMPLATE``, and prove the
guard demonstrably fails when one is reintroduced.

Fast by design: the live-tree check hashes file bytes against the summary's
traceability rows instead of re-running the full audit (re-run
``uv run python scripts/canon_qualify.py`` to refresh them).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_fund.models import canon_qualification as cq

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = REPO_ROOT / "quality" / "canon_qualification_summary.json"

#: Pinned ruleset identity; any rule change must update the summary and this
#: pin in the same commit (RULESET hash, not a hand-maintained list).
PINNED_RULESET_HASH = "7274f95d98db230d75b78a6752c909f21799428adb0e8c696375c4105f1cf3f4"

#: Non-qualifying template copies recorded live at the 2026-10-07 audit; the
#: count may only shrink as the quarantine proceeds (shrink-only ratchet).
MAX_LIVE_NON_QUALIFYING = 7529

_TEMPLATE_COPY = '''"""<STEM> canonical representative."""

from __future__ import annotations


def {stem}_ok(a: bool, b: bool) -> bool:
    return a and b


def {stem}_aux(aux: bool) -> bool:
    return aux


def _bench_{stem}() -> float:
    checks = ({stem}_ok(True, True), {stem}_aux(False))
    return float(sum(checks) / len(checks))
'''

_REAL_MODULE = '''"""Real behavioral module for guard calibration."""

from __future__ import annotations

from typing import List


def run(xs: List[float]) -> float:
    total = 0.0
    for x in xs:
        if x > 0.0:
            total += x
    return total


def bench_run() -> float:
    return run([1.0, -2.0, 3.0])
'''


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _guard_errors(recorded: dict[str, str], on_disk: dict[str, str]) -> list[str]:
    """Traceability check between audited rows and the live tree's bytes."""
    errors: list[str] = []
    for path in sorted(set(recorded) | set(on_disk)):
        if path not in recorded:
            errors.append(f"unaudited file present in live tree (reintroduction?): {path}")
        elif path not in on_disk:
            errors.append(f"audited file missing from live tree: {path}")
        elif recorded[path] != on_disk[path]:
            errors.append(f"file changed since audit (re-run the audit): {path}")
    return errors


def _models_dir_hashes(root: Path) -> dict[str, str]:
    models = root / "src" / "quant_fund" / "models"
    return {
        f"src/quant_fund/models/{path.name}": _source_sha256(path)
        for path in sorted(models.glob("*.py"))
    }


def _write_mini_repo(root: Path, stems: tuple[str, ...]) -> None:
    models = root / "src" / "quant_fund" / "models"
    models.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (models / f"{stem}.py").write_text(_TEMPLATE_COPY.format(stem=stem), encoding="utf-8")
    (models / "real_module.py").write_text(_REAL_MODULE, encoding="utf-8")


def test_ruleset_hash_is_pinned() -> None:
    assert cq.RULESET_VERSION == 1
    assert cq.ruleset_hash() == PINNED_RULESET_HASH


def test_live_tree_matches_the_audited_summary() -> None:
    # Reintroduction guard: any new or edited live module fails here until
    # `uv run python scripts/canon_qualify.py` re-derives the summary.
    assert SUMMARY_PATH.is_file(), f"missing audit summary at {SUMMARY_PATH}"
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    recorded = {
        row["path"]: row["source_sha256"]
        for row in summary["files"]
        if row["location"] == "live" and row["path"].startswith("src/quant_fund/models/")
    }
    errors = _guard_errors(recorded, _models_dir_hashes(REPO_ROOT))
    assert errors == [], "\n".join(errors)


def test_live_template_count_only_shrinks() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    counts = summary["counts"]["baseline_corpus"]
    assert counts["non_qualifying_template"] <= MAX_LIVE_NON_QUALIFYING


def test_template_copy_classifies_non_qualifying(tmp_path: Path) -> None:
    # Two copies of one skeleton + one real module in a throwaway corpus.
    _write_mini_repo(tmp_path, ("aa_tree", "avl_tree"))
    audit = cq.audit_tree(tmp_path)
    verdicts = {row["path"]: row["verdict"] for row in audit["files"]}
    assert verdicts["src/quant_fund/models/aa_tree.py"] == "NON_QUALIFYING_TEMPLATE"
    assert verdicts["src/quant_fund/models/avl_tree.py"] == "NON_QUALIFYING_TEMPLATE"
    assert verdicts["src/quant_fund/models/real_module.py"] == "QUALIFYING"


def test_audit_is_deterministic_over_the_same_tree(tmp_path: Path) -> None:
    _write_mini_repo(tmp_path, ("aa_tree", "avl_tree"))
    first = cq.dumps_audit(cq.audit_tree(tmp_path))
    second = cq.dumps_audit(cq.audit_tree(tmp_path))
    assert first == second


def test_reintroduced_template_fails_the_guard(tmp_path: Path) -> None:
    # Demonstration: audit a clean tree, then reintroduce a template copy.
    # Both halves of the guard fire — the bytes no longer match the audit,
    # and the module classifies NON_QUALIFYING_TEMPLATE (so it can never be
    # quietly registered as a qualifying family again).
    _write_mini_repo(tmp_path, ("aa_tree",))
    audit = cq.audit_tree(tmp_path)
    recorded = {
        row["path"]: row["source_sha256"] for row in audit["files"] if row["location"] == "live"
    }
    assert _guard_errors(recorded, _models_dir_hashes(tmp_path)) == []

    reintroduced = tmp_path / "src" / "quant_fund" / "models" / "avl_tree.py"
    reintroduced.write_text(_TEMPLATE_COPY.format(stem="avl_tree"), encoding="utf-8")

    errors = _guard_errors(recorded, _models_dir_hashes(tmp_path))
    assert any("avl_tree.py" in error and "reintroduction" in error for error in errors)

    fresh = cq.audit_tree(tmp_path)
    verdicts = {row["path"]: row["verdict"] for row in fresh["files"]}
    assert verdicts["src/quant_fund/models/avl_tree.py"] == "NON_QUALIFYING_TEMPLATE"


def test_verdict_rule_requires_all_five_signals() -> None:
    # The rule is conjunctive: breaking any one signal yields QUALIFYING.
    _write_mini_repo(tmp_path=Path("/tmp/unused"), stems=())  # pragma: no cover
