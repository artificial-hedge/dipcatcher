"""Selective family surface: eligible runners only, partial records, fail closed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_fund.research.catalog import registry
from quant_fund.research.family_surface import (
    RUN_SCHEMA,
    FamilySurfaceError,
    PanelBundle,
    family_contract,
    list_executable,
    parse_only,
    run_families,
    select_families,
)


def _runner_ids() -> set[str]:
    from quant_fund.research.family_surface import RUNNERS

    return set(RUNNERS)


def test_runners_are_exactly_the_required_families() -> None:
    assert _runner_ids() == set(registry.REQUIRED_BENCHMARK_FAMILIES)


def test_required_family_is_executable() -> None:
    contract = family_contract("volatility")
    assert contract["qualification"] == "required"
    assert contract["executable"] is True
    assert contract["runner"] == "research --only volatility"
    assert "qlike" in contract["scores"]


def test_retired_family_is_refused() -> None:
    retired = next(iter(registry.RETIRED_BENCHMARK_FAMILIES))
    contract = family_contract(retired)
    assert contract["qualification"] == "retired"
    assert contract["executable"] is False
    assert contract["refusal"]
    assert "historical receipts" in contract["refusal"]


def test_unknown_family_is_refused() -> None:
    contract = family_contract("not_a_real_family_zzz")
    assert contract["qualification"] == "unknown"
    assert contract["executable"] is False


def test_live_optional_without_runner_is_not_executable() -> None:
    live = registry.LIVE_OPTIONAL_BENCHMARK_FAMILIES - set(registry.REQUIRED_BENCHMARK_FAMILIES)
    family_id = "complexity" if "complexity" in live else sorted(live)[0]
    contract = family_contract(family_id)
    assert contract["qualification"] == "live_optional"
    assert contract["executable"] is False
    assert "selective runner" in contract["refusal"]


def test_unsafe_family_id_is_refused() -> None:
    with pytest.raises(FamilySurfaceError):
        family_contract("Volatility")
    with pytest.raises(FamilySurfaceError):
        parse_only("volatility;rm")
    with pytest.raises(FamilySurfaceError):
        parse_only("volatility,volatility")
    with pytest.raises(FamilySurfaceError):
        parse_only("")


def test_select_volatility_goal_and_exclude_retired_token() -> None:
    retired = next(iter(registry.RETIRED_BENCHMARK_FAMILIES))
    payload = select_families(
        f"score the volatility forecast with qlike and do not run {retired}",
        max_families=2,
    )
    ids = [row["id"] for row in payload["selected"]]
    assert ids == ["volatility"]
    assert all(row["executable"] is True for row in payload["selected"])
    assert any(row["id"] == retired for row in payload["exclusions"])


def test_select_empty_is_a_valid_no_suitable_family() -> None:
    payload = select_families("hello there", max_families=4)
    assert payload["selected"] == []
    assert payload["schema"] == "dipcatcher.family_surface.v1"


def test_select_respects_cap() -> None:
    payload = select_families(
        "volatility conformal jackknife ranking alpha drawdown",
        max_families=1,
    )
    assert len(payload["selected"]) == 1
    assert payload["exclusions"]


def test_negative_cap_refused() -> None:
    with pytest.raises(FamilySurfaceError):
        select_families("volatility", max_families=-1)


def test_list_executable_is_required_and_sorted() -> None:
    rows = list_executable()
    assert {row["id"] for row in rows} == set(registry.REQUIRED_BENCHMARK_FAMILIES)
    assert all(row["executable"] and row["qualification"] == "required" for row in rows)


def test_run_families_executes_only_the_named_runner(tmp_path: Path) -> None:
    seen: list[str] = []

    def cpcv() -> dict[str, float]:
        seen.append("cpcv")
        return {"n_folds": 2.0}

    def explode(*_args: object) -> None:
        raise AssertionError("panel loader must not run for a no-input family")

    config = SimpleNamespace(data=SimpleNamespace(root=tmp_path, source="synthetic"))
    payload = run_families(
        config,
        ["cpcv"],
        runners={"cpcv": cpcv},
        load_panel=explode,
    )
    assert seen == ["cpcv"]
    assert payload["schema"] == RUN_SCHEMA
    assert payload["partial"] is True
    assert payload["verified"] is False
    assert payload["evidence_class"] == "SYNTHETIC"
    assert payload["family_ids"] == ["cpcv"]
    assert len(payload["sha256"]) == 64
    artifact = Path(payload["artifact"])
    assert artifact.is_file()
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert "PARTIAL" in artifact.read_text(encoding="utf-8") or "partial" in artifact.read_text(
        encoding="utf-8"
    )


def test_run_families_refuses_retired_before_any_runner() -> None:
    retired = next(iter(registry.RETIRED_BENCHMARK_FAMILIES))
    called = False

    def runner() -> dict[str, str]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(FamilySurfaceError, match="retired"):
        run_families(
            SimpleNamespace(data=SimpleNamespace(source="market")),
            [retired],
            runners={retired: runner},
            write_artifact=False,
        )
    assert called is False


def test_run_families_refuses_when_estimate_exceeds_budget() -> None:
    def runner() -> dict[str, int]:
        raise AssertionError("budget must be checked before execution")

    with pytest.raises(FamilySurfaceError, match="budget_s"):
        run_families(
            SimpleNamespace(data=SimpleNamespace(source="synthetic")),
            ["cpcv"],
            runners={"cpcv": runner},
            write_artifact=False,
            budget_s=1,
        )


def test_run_families_does_not_replace_a_missing_panel() -> None:
    def loader(_config: object) -> PanelBundle:
        raise FileNotFoundError("market panel is not on disk")

    def runner(*_args: object) -> dict[str, str]:
        raise AssertionError("runner must not start after a panel failure")

    with pytest.raises(FileNotFoundError, match="market panel"):
        run_families(
            SimpleNamespace(data=SimpleNamespace(source="vendor-bars")),
            ["volatility"],
            runners={"volatility": runner},
            load_panel=loader,
            write_artifact=False,
        )


def test_run_families_rejects_duplicates_and_empty() -> None:
    with pytest.raises(FamilySurfaceError):
        run_families(SimpleNamespace(), [], write_artifact=False)
    with pytest.raises(FamilySurfaceError, match="duplicate"):
        run_families(SimpleNamespace(), ["cpcv", "cpcv"], write_artifact=False)
