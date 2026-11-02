"""Public lake, quality, and simulator entry points reject bad inputs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
import typer
from pydantic import ValidationError

from quant_fund.cli.lake_cmds import lake_quality
from quant_fund.data.lakehouse.lineage import format_dag, record_lineage, verify_lineage
from quant_fund.data.lakehouse.quality import QualityThresholds, quality_report
from quant_fund.data.lakehouse.store import write_partitioned_bars
from quant_fund.data.quality import QualityConfig, score_bars, score_datasets
from quant_fund.data.quality.cli import main as quality_cli
from quant_fund.market_sim.__main__ import _parser as market_sim_parser
from quant_fund.schemas.errors import DataContractError


def test_quality_report_rejects_empty_and_non_frames() -> None:
    with pytest.raises(DataContractError, match="non-empty"):
        quality_report(pl.DataFrame({"close": []}))
    with pytest.raises(DataContractError, match="DataFrame"):
        quality_report(None)  # type: ignore[arg-type]


def test_quality_thresholds_reject_bad_ranges() -> None:
    with pytest.raises(DataContractError, match="stale_run_length"):
        QualityThresholds(stale_run_length=1)
    with pytest.raises(DataContractError, match="max_abs_log_return"):
        QualityThresholds(max_abs_log_return=float("nan"))
    with pytest.raises(DataContractError, match="max_outliers"):
        QualityThresholds(max_outliers=-1)
    with pytest.raises(DataContractError, match="max_gap"):
        QualityThresholds(max_gap=timedelta(days=-1))


def test_score_bars_rejects_none_empty_and_bad_config() -> None:
    with pytest.raises(TypeError, match="got None"):
        score_bars(None)
    with pytest.raises(ValueError, match="non-empty"):
        score_bars(pl.DataFrame({"close": []}))
    with pytest.raises(ValidationError):
        QualityConfig(max_abs_log_return=float("inf"))
    with pytest.raises(ValidationError):
        QualityConfig(symbol_column="  ")
    with pytest.raises(ValueError, match="non-empty mapping"):
        score_datasets({})
    with pytest.raises(ValueError, match="dataset name"):
        score_datasets({"": pl.DataFrame({"close": [1.0]})})


def test_quality_cli_rejects_empty_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.parquet"
    pl.DataFrame({"close": []}).write_parquet(path)
    assert quality_cli([str(path)]) == 2
    assert "no rows" in capsys.readouterr().err


def test_lake_quality_cli_rejects_bad_thresholds() -> None:
    with pytest.raises(typer.BadParameter, match="max_abs_log_return"):
        lake_quality(
            path=Path("missing.parquet"),
            max_abs_log_return=float("nan"),
            max_gap_days=None,
            stale_run_length=5,
            enforce=True,
        )
    with pytest.raises(typer.BadParameter, match="stale_run_length"):
        lake_quality(
            path=Path("missing.parquet"),
            max_abs_log_return=0.5,
            max_gap_days=None,
            stale_run_length=0,
            enforce=True,
        )


def test_partitioned_bars_reject_empty_and_none(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="non-empty"):
        write_partitioned_bars(tmp_path, pl.DataFrame({"close": []}), dataset="bars")
    with pytest.raises(DataContractError, match="DataFrame"):
        write_partitioned_bars(tmp_path, None, dataset="bars")  # type: ignore[arg-type]


def test_lineage_rejects_blank_dataset(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="dataset name"):
        record_lineage(tmp_path, dataset=" ", snapshot_id="ab" * 32, code="x")
    with pytest.raises(DataContractError, match="dataset name"):
        format_dag(tmp_path, "")
    with pytest.raises(DataContractError, match="dataset name"):
        verify_lineage(tmp_path, "\n")


def test_market_sim_cli_rejects_bad_event_and_seed_counts() -> None:
    parser = market_sim_parser()
    with pytest.raises(SystemExit) as events:
        parser.parse_args(["bench", "--events", "0"])
    assert events.value.code == 2
    with pytest.raises(SystemExit) as seed:
        parser.parse_args(["bench", "--seed", "-1"])
    assert seed.value.code == 2
    with pytest.raises(SystemExit) as report_events:
        parser.parse_args(["report", "--events", "20000001"])
    assert report_events.value.code == 2
    parsed = parser.parse_args(["bench", "--events", "3", "--seed", "0"])
    assert parsed.events == 3
    assert parsed.seed == 0
