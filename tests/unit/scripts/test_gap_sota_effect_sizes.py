"""Tests for ``scripts/sota_effect_sizes.py``.

This script turns per-origin CRPS matrices into effect sizes with
stationary-bootstrap confidence intervals — magnitudes, not just p-values.
The properties that matter:

* a pair's delta is *target minus challenger*, so a positive mean means the
  challenger is better on that origin;
* a pair with too few complete observations is reported as
  ``insufficient_data`` instead of getting a fabricated confidence interval;
* only per-asset shards feed the pooled cell — the pooled matrix and
  multi-asset shards are excluded, or the pool would double-count;
* the receipt is honest about what it is: research output, no live-P&L
  claim.

Every input here is SYNTHETIC and constructed in the test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "sota_effect_sizes.py"
_spec = importlib.util.spec_from_file_location("gap_sota_effect_sizes", _SCRIPT_PATH)
assert _spec and _spec.loader, "sota_effect_sizes.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_sota_effect_sizes"] = mod
_spec.loader.exec_module(mod)

MODELS = ["kronos_small", "chronos2", "dip_a", "dip_b"]


def _matrix(n_origins: int = 40, offset: float = 0.5) -> np.ndarray:
    """A synthetic CRPS matrix: the challengers beat the targets by *offset*."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.5, size=(n_origins, len(MODELS)))
    matrix = base.copy()
    matrix[:, 2:] -= offset
    return matrix


# --------------------------------------------------------------------------- #
# Pair deltas
# --------------------------------------------------------------------------- #


def test_pair_deltas_are_target_minus_challenger() -> None:
    matrix = _matrix()
    labels, deltas = mod._pair_deltas(matrix, MODELS)
    assert labels == [
        "kronos_small_minus_dip_a",
        "kronos_small_minus_dip_b",
        "chronos2_minus_dip_a",
        "chronos2_minus_dip_b",
    ]
    # The challenger columns were lowered, so target minus challenger is positive.
    assert np.all(deltas.mean(axis=0) > 0.0)


def test_pair_deltas_have_one_column_per_pair() -> None:
    matrix = _matrix()
    _labels, deltas = mod._pair_deltas(matrix, MODELS)
    assert deltas.shape == (matrix.shape[0], 4)


def test_a_model_set_without_a_challenger_has_no_pair_columns() -> None:
    """No challenger means no pairs — and the helper refuses rather than guessing.

    ``numpy.stack`` rejects an empty list, so the script's precondition is
    that every shard carries at least one non-target model.
    """
    matrix = _matrix()
    with pytest.raises(ValueError):
        mod._pair_deltas(matrix, ["kronos_small", "chronos2"])


# --------------------------------------------------------------------------- #
# Effect rows
# --------------------------------------------------------------------------- #


def test_too_few_observations_report_insufficient_data() -> None:
    """No fabricated confidence interval on a thin sample."""
    deltas = np.ones((5, 1))
    rows = mod._effect_rows(
        "asset", ["kronos_small_minus_dip_a"], deltas, {"kronos_small": 1.0}, n_boot=10, seed=1
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "insufficient_data"
    assert rows[0]["n_origins"] == 5
    assert "ci95_lo" not in rows[0]


def test_a_usable_sample_produces_a_mean_and_a_confidence_interval() -> None:
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.5, 0.05, size=(60, 1))
    rows = mod._effect_rows(
        "asset", ["kronos_small_minus_dip_a"], deltas, {"kronos_small": 2.0}, n_boot=200, seed=3
    )
    row = rows[0]
    # A usable sample gets a real estimate, not the insufficient-data marker.
    assert "status" not in row
    assert row["n_origins"] == 60
    assert row["mean_delta"] == pytest.approx(float(deltas[:, 0].mean()))
    assert row["ci95_lo"] <= row["mean_delta"] <= row["ci95_hi"]
    assert row["significant_95"] is True
    assert row["block_len"] >= 1.0


def test_the_relative_improvement_uses_the_target_mean() -> None:
    deltas = np.full((60, 1), 0.5)
    rows = mod._effect_rows(
        "asset", ["kronos_small_minus_dip_a"], deltas, {"kronos_small": 2.0}, n_boot=50, seed=4
    )
    assert rows[0]["pct_vs_target_mean"] == pytest.approx(0.25)


def test_a_missing_target_mean_yields_no_relative_improvement() -> None:
    deltas = np.full((60, 1), 0.5)
    rows = mod._effect_rows("asset", ["kronos_small_minus_dip_a"], deltas, {}, n_boot=50, seed=4)
    assert rows[0]["pct_vs_target_mean"] is None


def test_non_finite_deltas_are_dropped_per_pair() -> None:
    deltas = np.full((60, 1), 0.5)
    deltas[:10, 0] = np.nan
    rows = mod._effect_rows(
        "asset", ["kronos_small_minus_dip_a"], deltas, {"kronos_small": 2.0}, n_boot=50, seed=4
    )
    assert rows[0]["n_origins"] == 50


# --------------------------------------------------------------------------- #
# Loading assets
# --------------------------------------------------------------------------- #


def _write_shard(
    directory: Path,
    name: str,
    models: list[str],
    assets: list[str],
    n_origins: int = 40,
) -> Path:
    path = directory / name
    np.savez(
        path,
        crps_matrix=_matrix(n_origins),
        model_names=np.asarray(models),
        meta_json=np.array(json.dumps({"asset_names": assets})),
    )
    return path


def test_only_single_asset_shards_are_loaded(tmp_path: Path) -> None:
    _write_shard(tmp_path, "d1_aapl.losses.npz", MODELS, ["aapl"])
    _write_shard(tmp_path, "d1_multi.losses.npz", MODELS, ["aapl", "msft"])
    assets = mod._load_assets(tmp_path, "d1")
    assert set(assets) == {"aapl"}


def test_the_pooled_merged_matrix_is_not_loaded(tmp_path: Path) -> None:
    _write_shard(tmp_path, "d1_aapl.losses.npz", MODELS, ["aapl"])
    _write_shard(tmp_path, "d1_merged.losses.npz", MODELS, ["aapl"])
    assets = mod._load_assets(tmp_path, "d1")
    assert set(assets) == {"aapl"}


def test_a_loaded_shard_carries_its_digest(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, "d1_aapl.losses.npz", MODELS, ["aapl"])
    assets = mod._load_assets(tmp_path, "d1")
    assert assets["aapl"]["sha"] == mod._sha256(path)
    assert assets["aapl"]["n_origins"] == 40


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def test_main_writes_an_honest_receipt(tmp_path: Path) -> None:
    _write_shard(tmp_path, "d1_aapl.losses.npz", MODELS, ["aapl"])
    out = tmp_path / "receipt.json"
    import sys as _sys

    argv = _sys.argv
    _sys.argv = [
        "sota_effect_sizes.py",
        "--eval-dir",
        str(tmp_path),
        "--horizon",
        "d1",
        "--n-boot",
        "50",
        "--out",
        str(out),
    ]
    try:
        assert mod.main() == 0
    finally:
        _sys.argv = argv
    receipt = json.loads(out.read_text())
    assert receipt["schema"] == "sota_effect_sizes.v1"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["pooled_summary"]["n_pairs"] > 0
    assert "aapl" in receipt["input_sha256"]


def test_main_fails_when_no_shards_are_present(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    import sys as _sys

    argv = _sys.argv
    _sys.argv = [
        "sota_effect_sizes.py",
        "--eval-dir",
        str(tmp_path),
        "--horizon",
        "d1",
        "--out",
        str(out),
    ]
    try:
        assert mod.main() == 1
    finally:
        _sys.argv = argv
    assert not out.exists()
