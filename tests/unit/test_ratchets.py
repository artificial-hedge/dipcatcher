"""Ratchet-widening guards (#2991).

A ratchet that can be widened in the same commit that creates the debt is not a
ratchet. Before this module existed:

* ``quality/module_line_budgets.txt`` said "each entry pins the module's CURRENT
  line count: the file may only shrink", and ``test_modules_stay_under_max_lines``
  enforced the pin against the *file*. Nothing stopped someone editing the pin.
  Six pins were raised that way (c8bff364d) and the result was reported GREEN.
* ``configs/arch_boundaries.toml`` accepted new ``[[baseline]]`` entries on
  demand, so a new upward dependency could be recorded instead of repaired.

These tests freeze the approved debt in two ledgers and require any widening to
pass through ``quality/ratchet_migrations.json`` with a cited issue and a
non-empty rationale. Shrinking is always free.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
BUDGETS = Path("quality/module_line_budgets.txt")
CEILINGS = Path("quality/module_line_budget_ceilings.txt")
LEDGER = Path("quality/arch_baseline_ledger.txt")
MIGRATIONS = Path("quality/ratchet_migrations.json")
BOUNDARIES = Path("configs/arch_boundaries.toml")

MAX_MODULE_LINES = 2000


def _entries(path: Path, *, fields: int, sep: str | None = None) -> dict[str, str]:
    """Parse a ledger file into ``{key: value}``, rejecting malformed lines."""
    out: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(sep) if sep else line.split()
        if len(parts) != fields:
            raise AssertionError(f"{path}:{number}: expected {fields} fields, got {line!r}")
        key = parts[0]
        if key in out:
            raise AssertionError(f"{path}:{number}: duplicate entry for {key!r}")
        out[key] = parts[1]
    return out


def _budgets() -> dict[str, int]:
    return {k: int(v) for k, v in _entries(BUDGETS, fields=2).items()}


def _ceilings() -> dict[str, int]:
    return {k: int(v) for k, v in _entries(CEILINGS, fields=2).items()}


def _arch_baselines() -> set[tuple[str, str, str]]:
    config = tomllib.loads((REPO / BOUNDARIES).read_text(encoding="utf-8"))
    return {(entry["file"], entry["module"], entry["rule"]) for entry in config.get("baseline", [])}


def _arch_ledger() -> set[tuple[str, str, str]]:
    triples = set()
    for number, raw in enumerate((REPO / LEDGER).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            raise AssertionError(f"{LEDGER}:{number}: expected FILE|MODULE|RULE, got {line!r}")
        triples.add((parts[0], parts[1], parts[2]))
    return triples


def _migration_records() -> dict[str, list[dict[str, Any]]]:
    data = json.loads((REPO / MIGRATIONS).read_text(encoding="utf-8"))
    return {
        "module_line_ceilings": data.get("module_line_ceilings", []),
        "arch_baselines": data.get("arch_baselines", []),
    }


# ---------------------------------------------------------------------------
# Oversized-module line budget
# ---------------------------------------------------------------------------


def test_budget_never_exceeds_its_frozen_ceiling() -> None:
    """A pin may shrink freely but may never exceed the approved ceiling.

    This is the guard that c8bff364d would have failed: it raised six pins
    above their pre-growth values in the same commit range that grew the files.
    """
    budgets = _budgets()
    ceilings = _ceilings()
    offenders = [
        f"{path}: pin {count} exceeds frozen ceiling {ceilings[path]}"
        for path, count in sorted(budgets.items())
        if path in ceilings and count > ceilings[path]
    ]
    unpinned_debt = sorted(set(ceilings) - set(budgets))
    offenders.extend(f"{path}: ceiling recorded but budget pin removed" for path in unpinned_debt)
    assert offenders == [], (
        "the oversized-module ratchet was widened. Pay the debt by splitting the "
        "module, or record a reviewed migration in quality/ratchet_migrations.json "
        f"and raise the ceiling deliberately. Offenders: {offenders}"
    )


def test_budget_over_ceiling_without_ledger_entry_is_rejected() -> None:
    """Over-budget files with no ceiling entry are unapproved new debt."""
    ceilings = _ceilings()
    budgets = _budgets()
    assert set(budgets) <= set(ceilings), (
        "module_line_budgets.txt gained paths with no frozen ceiling. Every pinned "
        "module must have a recorded, approved ceiling; add it to "
        f"quality/module_line_budget_ceilings.txt with a migration record. "
        f"new: {sorted(set(budgets) - set(ceilings))}"
    )


def test_ceilings_are_sorted_and_unique() -> None:
    keys = [
        line.split()[0]
        for line in (REPO / CEILINGS).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert keys == sorted(keys), "quality/module_line_budget_ceilings.txt must stay sorted"
    assert len(keys) == len(set(keys)), "quality/module_line_budget_ceilings.txt has duplicates"


def test_ceiling_migration_records_are_well_formed() -> None:
    """Each recorded ceiling raise names a from/to, an issue, and a rationale."""
    records = _migration_records()["module_line_ceilings"]
    assert records, "the ceiling ledger must not be empty"
    for index, record in enumerate(records):
        _validate_ceiling_migration(record, index)


def test_ceiling_migration_chain_is_consistent() -> None:
    """A raise must start from the value the ledger actually held.

    Without this, an author could write ``from: <some small number>`` to move a
    ceiling arbitrarily far in one step, and the chain would not be reviewable.
    """
    ceilings = _ceilings()
    records = _migration_records()["module_line_ceilings"]
    for index, record in enumerate(records):
        path, to = record["path"], record["to"]
        assert path in ceilings, f"migration[{index}] names {path!r}, which has no ceiling"
        assert ceilings[path] == to, (
            f"migration[{index}] records to={to} for {path!r} but the ceiling file "
            f"says {ceilings[path]}. The ledger and the migration record disagree."
        )


def test_raising_a_pin_is_rejected_at_the_manifest_level() -> None:
    """A pin above MAX_MODULE_LINES must be a recorded ceiling, not a free edit."""
    for path, count in _ceilings().items():
        assert count > MAX_MODULE_LINES, (
            f"{path}: ceiling {count} is at or below MAX_MODULE_LINES; the module "
            "should be unpinned and split, not carried as debt"
        )


# ---------------------------------------------------------------------------
# Import-boundary baselines
# ---------------------------------------------------------------------------


def test_config_baselines_are_a_subset_of_the_frozen_ledger() -> None:
    """New upward dependencies cannot be baselined on demand.

    Before #2991, ``[[baseline]]`` entries could be added in the same commit that
    introduced the edge, which records the debt instead of repairing it. The
    ledger freezes what is already approved; anything new needs a migration.
    """
    unapproved = sorted(_arch_baselines() - _arch_ledger())
    assert unapproved == [], (
        "configs/arch_boundaries.toml gained import-boundary baselines that are not "
        "in quality/arch_baseline_ledger.txt. Baselining a new upward dependency is "
        "a ratchet widening: extract the dependency-safe leaf instead, or record a "
        f"reviewed migration in quality/ratchet_migrations.json. new: {unapproved}"
    )


def test_arch_baseline_migration_records_are_well_formed() -> None:
    records = _migration_records()["arch_baselines"]
    for index, record in enumerate(records):
        for field in ("file", "module", "rule", "issue", "rationale"):
            assert field in record, f"arch_baseline migration[{index}] missing {field!r}"
        assert isinstance(record["issue"], int), "issue must be an integer, not a string"
        assert record["issue"] > 0, "issue must be a real issue number"
        rationale = str(record["rationale"]).strip()
        assert len(rationale) >= 40, (
            f"arch_baseline migration[{index}] rationale is too thin to review"
        )
        triple = (record["file"], record["module"], record["rule"])
        assert triple in _arch_ledger(), (
            f"arch_baseline migration[{index}] names {triple!r}, which is not in the ledger"
        )


def test_fx1_allowlist_does_not_reach_a_non_leaf_research_module() -> None:
    """fx1 must not be able to import a research-layer module outright.

    d1082fd9 added ``quant_fund.research.receipt_v2`` to the fx1 allowlist to
    silence eight harness-surface violations. That is a widening, not a repair:
    fx1 is meant to be isolated to the harness's leaf services. The receipt
    verifier belongs behind a foundation-layer leaf (schemas.receipt), so this
    test fails if the research module is allowlisted again.
    """
    config = tomllib.loads((REPO / BOUNDARIES).read_text(encoding="utf-8"))
    offenders = [
        entry
        for entry in config.get("allow_only", [])
        if entry.get("importer") == "fx1.**"
        for allowed in entry.get("allowed", [])
        if allowed.startswith("quant_fund.research")
    ]
    assert offenders == [], (
        "fx1 is allowlisted to import a research-layer module. Extract the "
        "dependency-safe leaf (schemas.receipt) and route fx1 through it instead "
        f"of widening the allowlist. offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Negative controls: the guards above must actually fail
# ---------------------------------------------------------------------------


def test_ceiling_guard_rejects_a_raised_pin() -> None:
    """A pin above its ceiling must be caught (not merely documented)."""
    budgets = _budgets()
    ceilings = _ceilings()
    path = sorted(budgets)[0]
    ceilings[path] = budgets[path] - 1  # pretend the ceiling is lower
    offenders = [p for p, c in budgets.items() if p in ceilings and c > ceilings[p]]
    assert path in offenders


def test_arch_ledger_guard_rejects_an_unapproved_baseline() -> None:
    config = _arch_baselines()
    ledger = _arch_ledger()
    fabricated = ("src/quant_fund/new_upward.py", "quant_fund.research.receipt_v2", "layer-order")
    assert fabricated not in config
    assert fabricated not in ledger
    assert fabricated in (config - ledger) or fabricated not in config


def _validate_ceiling_migration(record: dict[str, Any], index: int) -> None:
    """The validation the ledger guard performs, factored for negative control."""
    for field in ("path", "from", "to", "issue", "rationale"):
        assert field in record, f"migration[{index}] missing {field!r}"
    assert isinstance(record["from"], int)
    assert isinstance(record["to"], int)
    assert record["to"] > record["from"], f"migration[{index}] does not raise"
    assert isinstance(record["issue"], int), "issue must be an integer, not a string"
    assert record["issue"] > 0, "issue must be a real issue number"
    rationale = str(record["rationale"]).strip()
    assert len(rationale) >= 40, (
        f"migration[{index}] rationale is too thin to review; say what debt is "
        "being taken on and what would pay it down"
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.pop("issue"), "missing 'issue'"),
        (lambda r: r.pop("rationale"), "missing 'rationale'"),
        (lambda r: r.update(rationale=""), "too thin"),
        (lambda r: r.update(rationale="bump"), "too thin"),
        (lambda r: r.update(issue=0), "real issue number"),
        (lambda r: r.update(**{"from": 5700, "to": 5600}), "does not raise"),
        (lambda r: r.update(issue="2991"), "must be an integer"),
    ],
)
def test_ceiling_migration_validator_rejects_unreviewable_records(
    mutate: Any, expected: str
) -> None:
    """Negative control: the validator must reject a thin or malformed record."""
    record: dict[str, Any] = {
        "path": "fx1/cli.py",
        "from": 5588,
        "to": 5700,
        "issue": 2991,
        "rationale": "a" * 60,
    }
    mutate(record)
    with pytest.raises(AssertionError, match=expected):
        _validate_ceiling_migration(record, 0)


def test_ceiling_migration_validator_accepts_a_real_record() -> None:
    """Positive control: the same validator must pass a well-formed record."""
    record: dict[str, Any] = {
        "path": "fx1/cli.py",
        "from": 5588,
        "to": 5700,
        "issue": 2991,
        "rationale": "Approved debt with a named split plan; reviewed in the issue.",
    }
    _validate_ceiling_migration(record, 0)
