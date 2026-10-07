import numpy as np
import pytest

from quant_fund.decay.significance import (
    block_bootstrap_ic_pvalue,
    deflated_excess_t,
    expected_max_null_tstat,
)

pytestmark = pytest.mark.synthetic


def test_expected_max_null_single_trial_is_zero() -> None:
    assert expected_max_null_tstat(1) == 0.0


def test_expected_max_null_two_trials_matches_theory() -> None:
    # E[max(Z1, Z2)] = E|Z1 − Z2| / 2 = 1/√π
    assert expected_max_null_tstat(2) == pytest.approx(1.0 / np.sqrt(np.pi), abs=1e-6)


def test_expected_max_null_grows_with_trials() -> None:
    small = expected_max_null_tstat(10)
    large = expected_max_null_tstat(1000)
    assert large > small > 0.0


def test_deflated_excess_t_flags_best_of_many_nulls() -> None:
    rng = np.random.default_rng(60)
    n_trials = 200
    ic_std_se = 0.02
    # the best of 200 null ICs is close to the expected max — excess ≈ 0
    trials = rng.standard_normal(n_trials) * ic_std_se
    best = float(np.max(trials))
    t_obs = best / ic_std_se
    out = deflated_excess_t(t_obs, n_trials, ic_std_se)
    assert abs(out["excess_t"]) < 1.5


def test_deflated_excess_t_positive_for_real_signal() -> None:
    out = deflated_excess_t(t_obs=4.0, n_trials=50, ic_std_over_sqrt_t=0.01)
    assert out["excess_t"] > 0
    assert 0.0 < out["p_approx"] < 0.05
    assert out["excess_ic"] == pytest.approx(out["excess_t"] * 0.01)


def test_block_bootstrap_rejects_zero_mean_iid() -> None:
    rng = np.random.default_rng(61)
    ic = 0.12 + 0.2 * rng.standard_normal(400)
    out = block_bootstrap_ic_pvalue(ic, n_boot=500, block=5, seed=0)
    assert out["p"] < 0.05


def test_block_bootstrap_accepts_pure_noise() -> None:
    rng = np.random.default_rng(62)
    ic = 0.2 * rng.standard_normal(400)
    out = block_bootstrap_ic_pvalue(ic, n_boot=500, block=5, seed=0)
    assert out["p"] > 0.05


def test_block_bootstrap_pvalue_is_two_sided() -> None:
    rng = np.random.default_rng(63)
    ic = -0.12 + 0.2 * rng.standard_normal(400)
    out = block_bootstrap_ic_pvalue(ic, n_boot=500, block=5, seed=0)
    assert out["p"] < 0.05


def test_p_value_bounds() -> None:
    rng = np.random.default_rng(64)
    ic = rng.standard_normal(200)
    out = block_bootstrap_ic_pvalue(ic, n_boot=300, block=4, seed=1)
    assert 0.0 <= out["p"] <= 1.0
