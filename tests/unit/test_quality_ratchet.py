"""Ratchets for the harness quality bar.

The strict-module allowlist and the mccabe ceiling may tighten.
They must not shrink or rise. Per-function McCabe ceilings live in
``quality/mccabe_baseline.txt`` and are enforced by
``scripts/check_mccabe_ratchet.py``.
"""

from __future__ import annotations

import hashlib
import re
import runpy
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "quality" / "mypy_strict_modules.txt"
BASELINE = ROOT / "quality" / "mypy_strict_baseline.txt"
MCCABE_BASELINE = ROOT / "quality" / "mccabe_baseline.txt"
CHECKER = runpy.run_path(str(ROOT / "scripts" / "check_mypy_strict_allowlist.py"))
MCCABE = runpy.run_path(str(ROOT / "scripts" / "check_mccabe_ratchet.py"))
strict_entries = CHECKER["_entries"]
strict_allowlisted = CHECKER["allowlisted"]
mccabe_entries = MCCABE["_entries"]
# Initial set of strict-clean modules. Add to the allowlist; never remove these.
STRICT_MODULE_FLOOR = 668
STRICT_BASELINE_SHA256 = "452034ec90dbc11dc2a8ca78f22d950c591ae0fd67b3ecbfabe08d5906f7cdcd"
# validate_ledger_schema. verify_research_artifact was 196 before the split.
MCCABE_CEILING = 74
# Soft threshold + floor count for the per-function C901 ratchet. The
# floor may rise as debt is paid down via --write; it must not fall below
# this commit's recorded size without an intentional baseline rewrite.
MCCABE_SOFT_THRESHOLD = 10
MCCABE_BASELINE_FLOOR = 1
# `except Exception` handlers under src/quant_fund. Origin/main sat at 75;
# four closed lazy-import guards narrowed to ImportError (catalog ×3 +
# fast_replay forecast overlay), so the ceiling tightens to 71. The corpus-epoch
# integrity substrate adds seven deliberate fail-closed handlers (noqa: BLE001
# each): OTS explorer/calendar outages skip rather than fail, and verifier gates
# convert unexpected exceptions into recorded gate errors. Ceiling 78.
#
# The audit-lane merge wave (param_fuzz, asof_audit, flat_audit, cache_audit,
# map_parity, inherit_audit, boundary_audit, causality_scan, vine_dominance)
# adds fourteen more deliberate noqa: BLE001 handlers whose whole POINT is
# that unexpected exception classes are themselves audit findings — narrowing
# them would blind the audit. Deliberate baseline rewrite: ceiling 78 -> 92.
EXCEPT_EXCEPTION_CEILING = 92


def test_mypy_strict_allowlist_only_grows() -> None:
    lines = strict_entries(ALLOWLIST)
    baseline = strict_entries(BASELINE)
    assert lines == sorted(set(lines))
    assert baseline == sorted(set(baseline))
    assert hashlib.sha256("\n".join(baseline).encode()).hexdigest() == STRICT_BASELINE_SHA256
    assert set(baseline) <= set(lines)
    assert len(lines) >= STRICT_MODULE_FLOOR
    for line in lines:
        assert line.startswith("src/quant_fund/")
        assert line.endswith(".py")
        assert (ROOT / line).is_file(), line


def test_allowlist_parser_ignores_indented_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "src/quant_fund/sample.py"
    file = tmp_path / module
    file.parent.mkdir(parents=True)
    file.write_text("pass\n")
    listing = tmp_path / "allowlist.txt"
    listing.write_text(f"  # comment\n{module}\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(f"{module}\n")
    monkeypatch.setitem(strict_allowlisted.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(strict_allowlisted.__globals__, "ALLOWLIST", listing)
    monkeypatch.setitem(strict_allowlisted.__globals__, "BASELINE", baseline)
    assert strict_allowlisted() == [module]


def test_allowlist_rejects_dropped_baseline_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("src/quant_fund/pinned.py\n")
    listing = tmp_path / "allowlist.txt"
    listing.write_text("src/quant_fund/replacement.py\n")
    monkeypatch.setitem(strict_allowlisted.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(strict_allowlisted.__globals__, "ALLOWLIST", listing)
    monkeypatch.setitem(strict_allowlisted.__globals__, "BASELINE", baseline)
    with pytest.raises(SystemExit):
        strict_allowlisted()


def test_mccabe_ceiling_not_raised() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lint = cfg["tool"]["ruff"]["lint"]
    assert "C901" in lint["select"]
    assert lint["mccabe"]["max-complexity"] <= MCCABE_CEILING


def test_mccabe_per_function_baseline_shape() -> None:
    assert MCCABE["SOFT_THRESHOLD"] == MCCABE_SOFT_THRESHOLD
    rows = mccabe_entries(MCCABE_BASELINE)
    assert len(rows) >= MCCABE_BASELINE_FLOOR
    assert rows == dict(sorted(rows.items(), key=lambda item: (item[0][0], item[0][1])))
    for (path, name), complexity in rows.items():
        assert path.startswith("src/")
        assert path.endswith(".py")
        assert (ROOT / path).is_file(), path
        assert name
        assert complexity >= MCCABE_SOFT_THRESHOLD
    # Northset worst offender must stay on the ratchet (may only decrease).
    northset_bench = rows.get(("src/quant_fund/northset/benches.py", "bench_northset"))
    assert northset_bench is not None
    assert northset_bench <= 14


def test_mccabe_ratchet_rejects_raised_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "mccabe_baseline.txt"
    rows = {("src/quant_fund/sample.py", "too_complex"): 12}
    baseline.write_text(MCCABE["_format_baseline"](rows))
    monkeypatch.setitem(MCCABE["check"].__globals__, "BASELINE", baseline)
    monkeypatch.setitem(
        MCCABE["check"].__globals__,
        "_ruff_complexities",
        lambda _paths: {("src/quant_fund/sample.py", "too_complex"): 15},
    )
    assert MCCABE["check"]() == 1


def test_mccabe_write_refuses_raised_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "quality" / "mccabe_baseline.txt"
    baseline.parent.mkdir(parents=True)
    rows = {("src/quant_fund/sample.py", "too_complex"): 12}
    baseline.write_text(MCCABE["_format_baseline"](rows))
    monkeypatch.setitem(MCCABE["write_baseline"].__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(MCCABE["write_baseline"].__globals__, "BASELINE", baseline)
    monkeypatch.setitem(
        MCCABE["write_baseline"].__globals__,
        "_ruff_complexities",
        lambda _paths: {("src/quant_fund/sample.py", "too_complex"): 15},
    )
    assert MCCABE["write_baseline"]() == 1
    assert "12" in baseline.read_text()


def test_no_bare_except_and_exception_ceiling() -> None:
    bare = 0
    broad = 0
    for path in (ROOT / "src" / "quant_fund").rglob("*.py"):
        for line in path.read_text().splitlines():
            if re.match(r"\s*except\s*:", line):
                bare += 1
            elif re.match(r"\s*except\s+Exception\b", line):
                broad += 1
    assert bare == 0
    assert broad <= EXCEPT_EXCEPTION_CEILING


def test_type_ignore_manifest_matches_tree() -> None:
    import runpy

    module = runpy.run_path(str(ROOT / "scripts" / "check_type_ignores.py"))
    manifest = module["manifest_counts"]()
    actual = module["actual_counts"]()
    assert manifest == actual, (
        "quality/type_ignores.txt drifted; run `python scripts/update_type_ignores.py` "
        "and justify the new suppressions in review"
    )


def test_type_ignore_manifest_rejects_drift_and_dupes(tmp_path) -> None:
    import runpy

    module = runpy.run_path(str(ROOT / "scripts" / "check_type_ignores.py"))
    parser = module["manifest_counts"]
    bad = tmp_path / "type_ignores.txt"
    bad.write_text("src/x.py 1\nsrc/x.py 2\n")
    with pytest.raises(SystemExit):
        parser(bad)
    bad.write_text("src/x.py 0\n")
    with pytest.raises(SystemExit):
        parser(bad)
    bad.write_text("src/x.py notanint\n")
    with pytest.raises(SystemExit):
        parser(bad)
    good = tmp_path / "ok.txt"
    good.write_text("# comment\n\nsrc/x.py 3\nsrc/y.py 1\n")
    assert parser(good) == {"src/x.py": 3, "src/y.py": 1}


def test_broad_exception_manifest_matches_tree() -> None:
    """The global ceiling alone lets a new handler trade against an unrelated
    narrowing. The manifest pins the count per file so every broad catch is
    accounted to a place a reviewer can look at."""
    checker = runpy.run_path(str(ROOT / "scripts" / "check_broad_exceptions.py"))
    actual = checker["actual_counts"]()
    manifest = checker["manifest_counts"]()
    assert manifest == actual, (
        "broad-exception manifest drifted — regenerate with "
        "python scripts/update_broad_exceptions.py"
    )


def test_broad_exception_manifest_rejects_drift_and_dupes(tmp_path: Path) -> None:
    checker = runpy.run_path(str(ROOT / "scripts" / "check_broad_exceptions.py"))
    manifest_counts = checker["manifest_counts"]
    bad = tmp_path / "broad_exceptions.txt"
    bad.write_text("src/quant_fund/a.py 1\nsrc/quant_fund/a.py 2\n")
    with pytest.raises(SystemExit):
        manifest_counts(bad)
    bad.write_text("src/quant_fund/a.py 0\n")
    with pytest.raises(SystemExit):
        manifest_counts(bad)
    ok = tmp_path / "ok.txt"
    ok.write_text("# comment\nsrc/quant_fund/a.py 1\n")
    assert manifest_counts(ok) == {"src/quant_fund/a.py": 1}
