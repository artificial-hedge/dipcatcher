import numpy as np
import pytest

from quant_fund.combination.granger_ramanathan import (
    granger_ramanathan,
    rolling_combination,
)

pytestmark = pytest.mark.synthetic


def test_recovers_inverse_error_weights() -> None:
    rng = np.random.default_rng(90)
    n = 3000
    truth = rng.standard_normal(n)
    # member A: noise var 0.25; member B: noise var 1.0 (both unbiased)
    preds = np.column_stack([truth + 0.5 * rng.standard_normal(n), truth + rng.standard_normal(n)])
    out = granger_ramanathan(truth, preds, constrain=False)
    # unconstrained GR ≈ inverse-MSE weighting: w_A ≈ 0.8, w_B ≈ 0.2
    assert out["weights"][0] == pytest.approx(0.8, abs=0.15)
    assert out["weights"][1] == pytest.approx(0.2, abs=0.15)
    assert abs(float(out["intercept"])) < 0.1


def test_constrained_weights_on_simplex() -> None:
    rng = np.random.default_rng(91)
    n = 500
    truth = rng.standard_normal(n)
    preds = np.column_stack([truth + rng.standard_normal(n), truth + rng.standard_normal(n)])
    out = granger_ramanathan(truth, preds, constrain=True)
    assert np.all(out["weights"] >= 0.0)
    assert float(out["weights"].sum()) == pytest.approx(1.0)


def test_combined_beats_worst_member_in_sample() -> None:
    rng = np.random.default_rng(92)
    n = 1000
    truth = rng.standard_normal(n)
    good = truth + 0.1 * rng.standard_normal(n)
    bad = truth + 2.0 * rng.standard_normal(n)
    preds = np.column_stack([good, bad])
    out = granger_ramanathan(truth, preds, constrain=True)
    w = out["weights"]
    combined = preds @ w
    mse_comb = float(np.mean((truth - combined) ** 2))
    mse_bad = float(np.mean((truth - preds[:, 1]) ** 2))
    assert mse_comb < mse_bad


def test_rolling_combination_shapes_and_weights() -> None:
    rng = np.random.default_rng(93)
    n = 400
    truth = rng.standard_normal(n)
    preds = np.column_stack([truth + rng.standard_normal(n), truth + rng.standard_normal(n)])
    out = rolling_combination(truth, preds, window=100)
    assert out["combined"].shape == (300,)
    assert out["weights"].shape == (300, 2)
    assert np.all(np.abs(out["combined"]) < 10.0)


def test_rolling_combination_better_than_mean_member() -> None:
    rng = np.random.default_rng(94)
    n = 800
    truth = rng.standard_normal(n)
    member_a = truth + 0.3 * rng.standard_normal(n)
    member_b = truth + 1.2 * rng.standard_normal(n)
    preds = np.column_stack([member_a, member_b])
    out = rolling_combination(truth, preds, window=200)
    mse_comb = float(np.mean((truth[200:] - out["combined"]) ** 2))
    mse_mean = float(np.mean((truth[200:] - preds[200:].mean(axis=1)) ** 2))
    assert mse_comb < mse_mean


def test_validation() -> None:
    with pytest.raises(ValueError):
        granger_ramanathan(np.zeros(4), np.zeros((4, 3)))  # t <= k + 1
    with pytest.raises(ValueError):
        rolling_combination(np.zeros(10), np.zeros((10, 2)), window=3)
