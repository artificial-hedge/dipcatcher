import numpy as np
import pytest

from quant_fund.decay.bootstrap_ir import bootstrap_ir, bootstrap_ir_delta, information_ratio

pytestmark = pytest.mark.synthetic


def test_information_ratio_recovers_mu_over_sigma() -> None:
    rng = np.random.default_rng(60)
    ic = 0.05 + 0.1 * rng.standard_normal(500)
    ir = information_ratio(ic)
    assert ir == pytest.approx(0.5, abs=0.15)


def test_bootstrap_ir_ci_covers_truth() -> None:
    rng = np.random.default_rng(61)
    mu = 0.04
    ic = mu + 0.1 * rng.standard_normal(800)
    out = bootstrap_ir(ic, block=5, n_boot=500, seed=0)
    true_ir = mu / 0.1
    assert out["ci_lo"] <= true_ir <= out["ci_hi"]
    assert out["p_nonpositive"] < 0.05


def test_bootstrap_ir_se_finite() -> None:
    rng = np.random.default_rng(62)
    ic = rng.standard_normal(600)
    out = bootstrap_ir(ic, block=5, n_boot=400, seed=0)
    assert out["se"] > 0
    assert 0.0 <= out["p_nonpositive"] <= 1.0


def test_delta_interval_brackets_truth() -> None:
    rng = np.random.default_rng(63)
    mu = 0.05
    ic = mu + 0.12 * rng.standard_normal(2000)
    out = bootstrap_ir_delta(ic)
    true_ir = mu / 0.12
    assert out["ci_lo"] <= true_ir <= out["ci_hi"]
    assert out["se"] > 0


def test_validation() -> None:
    with pytest.raises(ValueError):
        information_ratio(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        bootstrap_ir(np.zeros(30), block=20)
