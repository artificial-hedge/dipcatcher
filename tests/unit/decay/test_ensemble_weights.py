import numpy as np
import pytest

from quant_fund.decay.ensemble_weights import (
    clip_negative_weights,
    combine_signals,
    decay_profile_weights,
    winsorized_ic_weights,
)

pytestmark = pytest.mark.synthetic


def test_winsorized_weights_favour_positive_ic() -> None:
    w = winsorized_ic_weights(np.array([0.10, 0.05, -0.20, 0.02]))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[3] > 0.0
    assert w[2] == 0.0


def test_winsorized_weights_all_negative_fall_back_equal() -> None:
    w = winsorized_ic_weights(np.array([-0.1, -0.2, -0.05]))
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0))


def test_winsorization_limits_outlier() -> None:
    rng = np.random.default_rng(70)
    ics = np.concatenate([rng.normal(0.03, 0.01, 9), [0.5]])
    w = winsorized_ic_weights(ics, limits=(0.0, 0.8))
    assert w[-1] < 0.5  # outlier cannot dominate


def test_clip_negative_renormalises() -> None:
    w = clip_negative_weights(np.array([-1.0, 1.0, 2.0]))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == 0.0
    assert w[2] == pytest.approx(2.0 / 3.0)


def test_decay_profile_weights_prefer_slow_strong_signals() -> None:
    curves = np.array(
        [
            [0.20, 0.10, 0.05, 0.02],  # fast decay
            [0.15, 0.13, 0.11, 0.09],  # slow decay
        ]
    )
    w = decay_profile_weights(curves, np.array([1.0, 8.0]))
    assert w[1] > w[0]
    assert w.sum() == pytest.approx(1.0)


def test_decay_profile_nan_half_life_falls_back() -> None:
    curves = np.array([[0.1, 0.1], [0.05, 0.05]])
    w = decay_profile_weights(curves, np.array([np.nan, 2.0]))
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)


def test_combine_signals_nan_robust() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[np.nan, 6.0], [7.0, np.nan]])
    out = combine_signals([a, b], np.array([0.5, 0.5]))
    assert out[0, 0] == 1.0  # only a available
    assert out[0, 1] == pytest.approx(4.0)  # mean of 2 and 6
    assert out[1, 0] == 5.0
    assert out[1, 1] == 4.0


def test_combine_signals_weights_uneven() -> None:
    a = np.array([[2.0, 4.0]])
    b = np.array([[6.0, 8.0]])
    out = combine_signals([a, b], np.array([0.75, 0.25]))
    assert out[0, 0] == pytest.approx(0.75 * 2.0 + 0.25 * 6.0)
    assert out[0, 1] == pytest.approx(0.75 * 4.0 + 0.25 * 8.0)


def test_combine_signals_all_nan_cell_is_nan() -> None:
    a = np.array([[np.nan, 1.0]])
    b = np.array([[np.nan, 3.0]])
    out = combine_signals([a, b], np.array([0.5, 0.5]))
    assert np.isnan(out[0, 0])
    assert out[0, 1] == 2.0
