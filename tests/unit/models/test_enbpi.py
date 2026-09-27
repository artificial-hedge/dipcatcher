"""Tests for EnbPI (quant_fund.models.enbpi). Seeded and deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.enbpi import EnbPI, coverage

Z_90 = 1.6448536269514722  # standard-normal two-sided 90% factor
LEVELS = np.array([0.05, 0.95])
SEED = 7
N_TRAIN = 700
N_TEST = 500


def _ar1_regime_change(
    seed: int, n_train: int = N_TRAIN, n_test: int = N_TEST
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) y_t = 0.6 y_{t-1} + sigma_t eps_t with a mid-test volatility jump.

    sigma_t = 1 before the test midpoint, 3 after — a heteroskedastic
    regime change that defeats fixed-width bands.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    eps = rng.standard_normal(n)
    sigma = np.where(np.arange(n) >= n_train + n_test // 2, 3.0, 1.0)
    y = np.empty(n, dtype=float)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + sigma[t] * eps[t]
    return y[:n_train], y[n_train:]


def _run_enbpi(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    adaptive: bool = True,
    phi_learning_rate: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit on y_train, stream y_test.

    Returns (grids, point forecasts, std of the post-fit training residual
    store) — the last is the width a frozen, unconformalized ensemble
    baseline calibrates to at training time.
    """
    model = EnbPI(
        LEVELS,
        n_bootstraps=15,
        random_state=seed,
        adaptive=adaptive,
        phi_learning_rate=phi_learning_rate,
    )
    model.fit(y_train)
    train_resid_std = float(np.std(model.residuals_))
    grids = np.stack([model.update(y) for y in y_test])
    return grids, model.forecast_history, train_resid_std


def _trailing_coverage(grids: np.ndarray, y: np.ndarray, window: int = 200) -> float:
    hit = (y >= grids[:, 0]) & (y <= grids[:, -1])
    return float(np.mean(hit[-window:]))


def test_trailing_coverage_within_tolerance_of_nominal() -> None:
    y_train, y_test = _ar1_regime_change(SEED)
    grids, _, _ = _run_enbpi(y_train, y_test)
    cov = _trailing_coverage(grids, y_test)
    assert abs(cov - 0.90) <= 0.08, f"trailing-200 coverage {cov:.3f} far from 0.90"


def test_adaptive_phi_at_least_as_close_as_frozen() -> None:
    y_train, y_test = _ar1_regime_change(SEED)
    grids_ad, _, _ = _run_enbpi(y_train, y_test, adaptive=True)
    grids_fr, _, _ = _run_enbpi(y_train, y_test, adaptive=False)
    err_ad = abs(_trailing_coverage(grids_ad, y_test, window=N_TEST) - 0.90)
    err_fr = abs(_trailing_coverage(grids_fr, y_test, window=N_TEST) - 0.90)
    assert err_ad <= err_fr, f"adaptive coverage error {err_ad:.3f} worse than frozen {err_fr:.3f}"


def test_grids_non_crossing_every_step() -> None:
    y_train, y_test = _ar1_regime_change(SEED)
    grids, _, _ = _run_enbpi(y_train, y_test)
    assert np.all(np.isfinite(grids))
    assert bool(np.all(np.diff(grids, axis=1) >= 0.0))


def test_beats_raw_ensemble_by_clear_margin() -> None:
    y_train, y_test = _ar1_regime_change(SEED)
    grids, centers, train_resid_std = _run_enbpi(y_train, y_test)
    raw_lo = centers - Z_90 * train_resid_std
    raw_hi = centers + Z_90 * train_resid_std
    raw_cov = float(np.mean((y_test >= raw_lo) & (y_test <= raw_hi)))
    enbpi_cov = float(np.mean((y_test >= grids[:, 0]) & (y_test <= grids[:, -1])))
    assert abs(enbpi_cov - 0.90) < abs(raw_cov - 0.90) - 0.05, (
        f"EnbPI {enbpi_cov:.3f} did not clearly beat raw ensemble {raw_cov:.3f}"
    )


def test_seeded_runs_are_deterministic() -> None:
    y_train, y_test = _ar1_regime_change(SEED)
    grids_a, centers_a, _ = _run_enbpi(y_train, y_test)
    grids_b, centers_b, _ = _run_enbpi(y_train, y_test)
    np.testing.assert_array_equal(grids_a, grids_b)
    np.testing.assert_array_equal(centers_a, centers_b)


def test_coverage_helper() -> None:
    grid = np.array([1.0, 2.0, 3.0])
    assert coverage(grid, 2.5) == 1.0
    assert coverage(grid, 3.5) == 0.0
    with pytest.raises(ValueError):
        coverage(np.array([]), 1.0)
    with pytest.raises(ValueError):
        coverage(grid, np.nan)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        EnbPI(np.array([0.95, 0.05]))  # not increasing
    with pytest.raises(ValueError):
        EnbPI(np.array([0.0, 0.9]))  # outside (0, 1)
    with pytest.raises(ValueError):
        EnbPI(np.empty(0))  # empty levels
    with pytest.raises(ValueError):
        EnbPI(np.array([0.95, 0.95]))  # not strictly increasing
    with pytest.raises(ValueError):
        EnbPI(LEVELS, n_bootstraps=0)
    with pytest.raises(ValueError):
        EnbPI(LEVELS, block_length=0)
    with pytest.raises(ValueError):
        EnbPI(LEVELS, phi_init=0.0)
    with pytest.raises(ValueError):
        EnbPI(LEVELS, phi_learning_rate=-0.1)


def test_fail_closed_fit_inputs() -> None:
    model = EnbPI(LEVELS, n_bootstraps=5, random_state=0)
    with pytest.raises(ValueError):
        model.fit(np.empty(0))
    with pytest.raises(ValueError):
        model.fit(np.array([0.1, np.nan, 0.3, 0.4, 0.5]))
    with pytest.raises(ValueError):
        model.fit(np.arange(3.0))  # length <= lags


def test_update_before_fit_raises_runtime_error() -> None:
    model = EnbPI(LEVELS, n_bootstraps=5, random_state=0)
    with pytest.raises(RuntimeError):
        model.update(1.0)


def test_update_validates_inputs() -> None:
    rng = np.random.default_rng(1)
    y_train = rng.standard_normal(120)
    model = EnbPI(LEVELS, n_bootstraps=5, random_state=0)
    model.fit(y_train)
    with pytest.raises(ValueError):
        model.update(np.array([1.0, 2.0]))  # non-scalar y
    with pytest.raises(ValueError):
        model.update(np.nan)
    with pytest.raises(ValueError):
        model.update(1.0, x=np.array([1.0, 2.0]))  # wrong feature length (lags=3)
    with pytest.raises(ValueError):
        model.update(1.0, x=np.array([np.nan, 1.0, 1.0]))
