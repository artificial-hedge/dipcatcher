"""CLI bridge for the real-data benchmark lanes.

``real_benchmark``, ``net_tournament`` and ``ranker_probability`` were the last
research lanes with an operator-facing entry point that ``dipcatcher --help``
did not show. An import-closure check counted them as wired — ``phase1_verify``
imports ``net_tournament`` purely to hash it for receipt provenance — while the
only way to actually run them was ``python -m``.

These tests drive the full freeze-then-score workflow through the console
script: a benchmark protocol is frozen, scored on validation, then handed to a
net-of-cost tournament that is itself frozen and run. That ordering is the
point of the lanes, so the tests assert it — a test phase before validation
must fail closed, and the published documents must carry the honesty flags.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app

runner = CliRunner()

_NAMES = ("A", "B", "C")
_N_DAYS = 240


def _bars(tmp_path: Path) -> Path:
    """A generated three-security daily panel with close/open/volume."""
    rng = np.random.default_rng(88)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i) for i in range(_N_DAYS)]
    frames = []
    for index, name in enumerate(_NAMES):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, _N_DAYS)))
        opening = np.empty(_N_DAYS)
        opening[0] = close[0]
        opening[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, _N_DAYS - 1))
        frames.append(
            pl.DataFrame(
                {
                    "security_id": [name] * _N_DAYS,
                    "event_time": dates,
                    "available_time": dates,
                    "ingested_time": dates,
                    "source": ["generated_contract_test"] * _N_DAYS,
                    "close": close + index,
                    "open": opening + index,
                    "volume": rng.integers(100_000, 900_000, _N_DAYS).astype(float),
                }
            )
        )
    path = tmp_path / "bars.parquet"
    pl.concat(frames).write_parquet(path)
    return path


def _protocol(bars: Path) -> dict:
    return {
        "dataset_path": str(bars),
        "dataset_sha256": hashlib.sha256(bars.read_bytes()).hexdigest(),
        "source_url": "https://example.com/generated-contract-test",
        "usage_basis": "generated contract test",
        "price_column": "close",
        "price_adjustment": "no actions in fixture",
        "universe_description": "three generated securities",
        "survivorship_bias": True,
        "availability_basis": "reconstructed",
        "holdout_previously_inspected": True,
        "train_start": "2020-01-01",
        "train_end": "2020-03-31",
        "validation_start": "2020-04-05",
        "validation_end": "2020-05-31",
        "test_start": "2020-06-05",
        "test_end": "2020-08-20",
        "min_train_rows": 100,
        "min_score_dates": 20,
    }


def _tournament_spec() -> dict:
    return {
        "execution": {},
        "trials": [
            {"name": "mom", "family": "momentum"},
            {"name": "rev", "family": "reversal", "lookback": 1},
        ],
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "open_column": "open",
        "volume_column": "volume",
        "price_basis": "raw_price_return",
        "n_boot": 99,
        "block_sessions": 5,
        "seed": 17,
    }


@pytest.fixture
def frozen_benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """A benchmark run frozen through the CLI. Returns (run_dir, protocol_path)."""
    bars = _bars(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(bars)))
    run = tmp_path / "benchmark"

    prepared = runner.invoke(
        app,
        [
            "real-benchmark",
            "prepare",
            "--protocol",
            str(protocol_path),
            "--output",
            str(run),
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    return run, protocol_path


@pytest.mark.parametrize("group", ["real-benchmark", "net-tournament", "ranker-probability"])
def test_group_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0, result.output


def test_real_benchmark_prepare_freezes_without_publishing_scores(
    tmp_path: Path, frozen_benchmark: tuple[Path, Path]
) -> None:
    run, _protocol_path = frozen_benchmark
    assert (run / "manifest.json").is_file()
    manifest = json.loads((run / "manifest.json").read_text())
    # The freeze step must not leak a score — that is the whole point of
    # freezing before scoring.
    assert "scores" not in manifest
    assert manifest["holdout_status"] == "previously_inspected"
    assert manifest["live_pnl_claim"] is False
    assert manifest["limitations"]


def test_real_benchmark_prepare_labels_the_data(tmp_path: Path) -> None:
    bars = _bars(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(bars)))
    result = runner.invoke(
        app,
        [
            "real-benchmark",
            "prepare",
            "--protocol",
            str(protocol_path),
            "--output",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    # REAL inputs must never be labelled SYNTHETIC.
    assert f"DATA_LABEL={protocol_path.name}" in result.output
    assert "DATA_LABEL=SYNTHETIC" not in result.output


def test_real_benchmark_score_validation_then_test(
    frozen_benchmark: tuple[Path, Path],
) -> None:
    run, _ = frozen_benchmark

    test_first = runner.invoke(
        app, ["real-benchmark", "score", "--run", str(run), "--phase", "test"]
    )
    assert test_first.exit_code != 0, "test phase must require validation first"

    validation = runner.invoke(
        app, ["real-benchmark", "score", "--run", str(run), "--phase", "validation"]
    )
    assert validation.exit_code == 0, validation.output
    assert f"DATA_LABEL={run.name}" in validation.output

    scored = runner.invoke(app, ["real-benchmark", "score", "--run", str(run), "--phase", "test"])
    assert scored.exit_code == 0, scored.output
    # Heavy frozen inputs stay out of stdout; the verdict is what an operator reads.
    assert "benchmark_manifest" not in scored.output


@pytest.mark.parametrize("phase", ["nonsense", ""])
def test_real_benchmark_rejects_a_bad_phase(
    frozen_benchmark: tuple[Path, Path], phase: str
) -> None:
    run, _ = frozen_benchmark
    result = runner.invoke(app, ["real-benchmark", "score", "--run", str(run), "--phase", phase])
    assert result.exit_code != 0
    assert "validation or test" in result.output


def test_real_benchmark_rejects_a_bad_receipt_version(tmp_path: Path) -> None:
    bars = _bars(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(bars)))
    result = runner.invoke(
        app,
        [
            "real-benchmark",
            "prepare",
            "--protocol",
            str(protocol_path),
            "--output",
            str(tmp_path / "run"),
            "--receipt-version",
            "3",
        ],
    )
    assert result.exit_code != 0
    assert "--receipt-version must be 1 or 2" in result.output


def test_net_tournament_freeze_then_run(
    tmp_path: Path, frozen_benchmark: tuple[Path, Path]
) -> None:
    benchmark_run, _ = frozen_benchmark
    spec_path = tmp_path / "slate.json"
    spec_path.write_text(json.dumps(_tournament_spec()))
    destination = tmp_path / "tournament"

    prepared = runner.invoke(
        app,
        [
            "net-tournament",
            "prepare",
            "--benchmark-run",
            str(benchmark_run),
            "--spec",
            str(spec_path),
            "--output",
            str(destination),
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    assert f"DATA_LABEL={spec_path.name}" in prepared.output
    assert (destination / "manifest.json").is_file()

    ran = runner.invoke(
        app, ["net-tournament", "run", "--run", str(destination), "--phase", "validation"]
    )
    assert ran.exit_code == 0, ran.output
    assert f"DATA_LABEL={destination.name}" in ran.output
    # The bulky frozen scenario table is stripped from stdout, as the argparse
    # main does, so the two surfaces print the same thing.
    assert "scenarios" not in ran.output


def test_net_tournament_run_rejects_a_bad_phase(
    tmp_path: Path, frozen_benchmark: tuple[Path, Path]
) -> None:
    benchmark_run, _ = frozen_benchmark
    spec_path = tmp_path / "slate.json"
    spec_path.write_text(json.dumps(_tournament_spec()))
    destination = tmp_path / "tournament"
    prepared = runner.invoke(
        app,
        [
            "net-tournament",
            "prepare",
            "--benchmark-run",
            str(benchmark_run),
            "--spec",
            str(spec_path),
            "--output",
            str(destination),
        ],
    )
    assert prepared.exit_code == 0, prepared.output

    result = runner.invoke(
        app, ["net-tournament", "run", "--run", str(destination), "--phase", "later"]
    )
    assert result.exit_code != 0
    assert "validation or test" in result.output


def test_ranker_probability_rejects_incomplete_input_selection(tmp_path: Path) -> None:
    """The two input modes are mutually exclusive and each needs its pair."""
    out = tmp_path / "receipt.json"

    neither = runner.invoke(app, ["ranker-probability", "run", "--output", str(out)])
    assert neither.exit_code != 0
    assert "--features and --labels" in neither.output

    only_features = tmp_path / "features.parquet"
    pl.DataFrame({"a": [1.0]}).write_parquet(only_features)
    half = runner.invoke(
        app,
        [
            "ranker-probability",
            "run",
            "--output",
            str(out),
            "--features",
            str(only_features),
        ],
    )
    assert half.exit_code != 0
    assert "--features and --labels" in half.output


def test_ranker_probability_rejects_a_half_specified_bronze_build(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    out = tmp_path / "receipt.json"

    no_lake = runner.invoke(
        app,
        [
            "ranker-probability",
            "run",
            "--output",
            str(out),
            "--bronze-root",
            str(bronze),
        ],
    )
    assert no_lake.exit_code != 0
    assert "--bronze-root requires --lake-root" in no_lake.output

    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    pl.DataFrame({"a": [1.0]}).write_parquet(features)
    pl.DataFrame({"a": [1.0]}).write_parquet(labels)
    mixed = runner.invoke(
        app,
        [
            "ranker-probability",
            "run",
            "--output",
            str(out),
            "--bronze-root",
            str(bronze),
            "--lake-root",
            str(tmp_path / "lake"),
            "--features",
            str(features),
            "--labels",
            str(labels),
        ],
    )
    assert mixed.exit_code != 0
    assert "excludes --features/--labels" in mixed.output


def test_ranker_probability_rejects_a_bad_receipt_version(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    pl.DataFrame({"a": [1.0]}).write_parquet(features)
    pl.DataFrame({"a": [1.0]}).write_parquet(labels)
    result = runner.invoke(
        app,
        [
            "ranker-probability",
            "run",
            "--output",
            str(tmp_path / "receipt.json"),
            "--features",
            str(features),
            "--labels",
            str(labels),
            "--receipt-version",
            "9",
        ],
    )
    assert result.exit_code != 0
    assert "--receipt-version must be 1 or 2" in result.output
