import numpy as np
import pytest

from quant_fund.combination.qlike_optimal import (
    crps_weights,
    min_var_weights,
    shrink_to_equal,
)

pytestmark = pytest.mark.synthetic


def test_min_var_two_members_known_variances() -> None:
    rng = np.random.default_rng(100)
    n = 20000
    # independent member errors with variances 0.25 and 1.0 → optimal
    # combination weights are proportional to inverse variances: 0.8 / 0.2
    e = np.column_stack([0.5 * rng.standard_normal(n), rng.standard_normal(n)])
    w = min_var_weights(e)
    assert w[0] == pytest.approx(0.8, abs=0.03)
    assert w[1] == pytest.approx(0.2, abs=0.03)


def test_min_var_reduces_error_variance() -> None:
    rng = np.random.default_rng(101)
    n = 4000
    e = rng.standard_normal((n, 3)) @ np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
    w = min_var_weights(e)
    combined = e @ w
    var_comb = float(np.var(combined))
    var_member = float(np.mean(np.var(e, axis=0)))
    assert var_comb < var_member


def test_min_var_weights_sum_to_one() -> None:
    rng = np.random.default_rng(102)
    e = rng.standard_normal((500, 4))
    w = min_var_weights(e)
    assert float(w.sum()) == pytest.approx(1.0)


def test_shrink_to_equal_endpoints() -> None:
    w = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(shrink_to_equal(w, 1.0), w)
    np.testing.assert_allclose(shrink_to_equal(w, 0.0), np.full(3, 1.0 / 3.0))
    half = shrink_to_equal(w, 0.5)
    assert half[0] == pytest.approx(0.5 + 1.0 / 6.0)


def test_crps_weights_concentrate_on_better_member() -> None:
    scores = np.array([0.10, 0.50])  # member 0 much better
    w = crps_weights(scores, temperature=0.1)
    assert w[0] == pytest.approx(1.0 / (1.0 + np.exp(-4.0)), abs=1e-9)
    assert w[0] > 0.97


def test_crps_weights_temperature_extremes() -> None:
    scores = np.array([0.1, 0.2])
    hot = crps_weights(scores, temperature=1000.0)
    np.testing.assert_allclose(hot, [0.5, 0.5], atol=1e-3)
    cold = crps_weights(scores, temperature=0.01)
    assert cold[0] > 0.999


def test_validation() -> None:
    with pytest.raises(ValueError):
        min_var_weights(np.zeros((2, 5)))
    with pytest.raises(ValueError):
        crps_weights(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        shrink_to_equal(np.zeros(3), 1.5)
