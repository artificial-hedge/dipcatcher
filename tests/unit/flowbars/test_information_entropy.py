import numpy as np
import pytest

from quant_fund.flowbars.entropy import (
    entropy_weights,
    rolling_shannon_entropy,
    shannon_entropy,
)
from quant_fund.flowbars.information import (
    cumulative_quadratic_variation,
    information_bar_ids,
)

pytestmark = pytest.mark.synthetic


def test_shannon_entropy_constant_is_zero() -> None:
    out = shannon_entropy(np.full(200, 3.14), bins=10)
    assert out["nats"] == 0.0
    assert out["normalised"] == 0.0


def test_shannon_entropy_uniform_is_max() -> None:
    x = np.linspace(-1, 1, 10000)
    out = shannon_entropy(x, bins=10)
    assert out["normalised"] == pytest.approx(1.0, abs=0.01)


def test_shannon_entropy_uniform_exceeds_normal() -> None:
    rng = np.random.default_rng(0)
    uniform = rng.uniform(-1, 1, 20000)
    normal = rng.standard_normal(20000)
    assert shannon_entropy(uniform, bins=10)["nats"] > shannon_entropy(normal, bins=10)["nats"]


def test_rolling_entropy_shape_and_nan_head() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    out = rolling_shannon_entropy(x, window=10, bins=4)
    assert out.shape == (100,)
    assert np.all(np.isnan(out[:9]))
    assert np.all(np.isfinite(out[9:]))


def test_entropy_weights_normalised() -> None:
    w = entropy_weights(np.array([0.5, 1.0, 0.25]))
    assert w.sum() == pytest.approx(1.0)
    assert w[1] > w[0] > w[2]


def test_entropy_weights_all_zero_falls_back_uniform() -> None:
    w = entropy_weights(np.array([0.0, 0.0]))
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_information_bar_ids_threshold() -> None:
    prices = 100.0 * np.exp(np.array([0.0, 0.1, 0.1, 0.3, 0.3, 0.9]))
    ids = information_bar_ids(prices, threshold=0.02)
    # running squared log-return: 0.01 (i=1), 0.01 (i=2), 0.05 → close at
    # i=3 (bar 1), 0 (i=4), 0.36 → close at i=5 (bar 2)
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 2])


def test_cumulative_quadratic_variation() -> None:
    prices = 100.0 * np.exp(np.array([0.0, 0.1, 0.2]))
    qv = cumulative_quadratic_variation(prices)
    assert qv[0] == 0.0
    assert qv[-1] == pytest.approx(0.01 + 0.01)


def test_information_bar_ids_rejects_nonpositive_prices() -> None:
    with pytest.raises(ValueError):
        information_bar_ids(np.array([100.0, -1.0]), 0.1)
