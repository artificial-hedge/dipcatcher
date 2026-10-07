"""Unit tests for quant_fund.models.lee_carter."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lee_carter import bench_lee_carter, lc_fit, lc_forecast


def _panel(n_a: int = 8, n_t: int = 40) -> np.ndarray:
    rng = np.random.default_rng(5)
    ages = np.arange(n_a, dtype=float)
    ax = -4.0 + 0.25 * ages
    bx = np.exp(-0.5 * ((ages - ages.mean()) / 2.5) ** 2)
    bx /= bx.sum()
    kt = 30.0 + 0.8 * np.arange(n_t)
    return ax[:, None] + np.outer(bx, kt) + rng.standard_normal((n_a, n_t)) * 0.05


def test_shapes_and_normalization() -> None:
    fit = lc_fit(_panel())
    assert fit["ax"].shape == (8,)
    assert fit["bx"].shape == (8,)
    assert fit["kt"].shape == (40,)
    assert abs(float(fit["bx"].sum()) - 1.0) < 1e-9


def test_forecast_shape_and_direction() -> None:
    fit = lc_fit(_panel())
    f3 = lc_forecast(fit, 3)
    assert f3.shape == (8,)
    # Positive drift -> higher log-mortality than last observation.
    assert float(f3.mean()) > float(fit["ax"].mean() + fit["bx"].mean() * fit["kt"][-1])


def test_reconstruction_quality() -> None:
    fit = lc_fit(_panel())
    assert float(fit["recon_r2"][0]) > 0.98


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        lc_fit(np.ones((2, 40)))
    with pytest.raises(ValueError):
        lc_fit(np.full((8, 40), np.nan))
    with pytest.raises(ValueError):
        lc_forecast(lc_fit(_panel()), 0)


def test_bench_contract() -> None:
    out = bench_lee_carter()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_lc_rmse"] < out["synthetic_lc_rmse_naive"]
