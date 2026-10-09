import numpy as np
import pytest

from quant_fund.flowbars.realized import (
    bipower_variation,
    bns_ratio_test,
    huang_tauchen_z,
    integrated_jump_component,
    medianrv,
    minrv,
    realized_kernel,
    realized_variance,
)

pytestmark = pytest.mark.synthetic

N_GRID = 8000
GRID_VOL = 0.01
IV = N_GRID * GRID_VOL * GRID_VOL  # 0.8


def _gbm(n: int, vol: float, seed: int, n_jumps: int = 0, jump_size: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol, n)
    if n_jumps:
        idx = rng.choice(n, size=n_jumps, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_jumps)
        r[idx] += signs * jump_size * vol
    return np.exp(np.concatenate(([0.0], np.cumsum(r))))


def test_rv_recovers_integrated_variance() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=1)
    assert realized_variance(p) == pytest.approx(IV, rel=0.1)


def test_bv_minrv_medianrv_robust_estimators() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=2)
    assert bipower_variation(p) == pytest.approx(IV, rel=0.15)
    assert minrv(p) == pytest.approx(IV, rel=0.15)
    assert medianrv(p) == pytest.approx(IV, rel=0.2)


def test_realized_kernel_close_to_rv_on_clean_data() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=3)
    rk = realized_kernel(p, bandwidth=50)
    rv = realized_variance(p)
    assert rk > 0
    assert abs(rk - rv) < 0.5 * rv


def test_realized_kernel_removes_microstructure_noise() -> None:
    rng = np.random.default_rng(6)
    p_eff = _gbm(N_GRID, GRID_VOL, seed=7)
    noise = rng.standard_normal(N_GRID + 1) * GRID_VOL * 0.5
    p_obs = p_eff * np.exp(noise)
    rv = realized_variance(p_obs)
    rk = realized_kernel(p_obs, bandwidth=50)
    assert rv > IV * 1.2  # noise inflates RV
    assert abs(rk - IV) < abs(rv - IV)  # kernel cancels the noise bias


def test_jumps_inflate_rv_over_bv() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=4, n_jumps=80, jump_size=5.0)
    rv = realized_variance(p)
    bv = bipower_variation(p)
    assert rv > bv + 0.08
    assert integrated_jump_component(p) > 0.05


def test_jump_tests_reject_with_jumps() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=5, n_jumps=80, jump_size=5.0)
    out_ratio = bns_ratio_test(p, n_boot=400, seed=0)
    out_z = huang_tauchen_z(p, n_boot=400, seed=0)
    assert out_ratio["ratio"] > 1.1
    assert out_ratio["p"] < 0.05
    assert out_z["p"] < 0.05


def test_jump_tests_accept_without_jumps() -> None:
    p = _gbm(N_GRID, GRID_VOL, seed=6)
    out_ratio = bns_ratio_test(p, n_boot=300, seed=0)
    out_z = huang_tauchen_z(p, n_boot=300, seed=0)
    assert out_ratio["p"] > 0.02
    assert out_z["p"] > 0.02


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        realized_variance(np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        realized_variance(np.array([1.0, 2.0]))
