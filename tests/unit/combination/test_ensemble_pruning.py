import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.ensemble_pruning import greedy_prune, pruned_weights, pruning_gain

pytestmark = pytest.mark.synthetic

TAUS = np.linspace(0.1, 0.9, 9)


def _members(seed: int, n: int = 2000):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    z = norm.ppf(TAUS)
    good = y[:, None] + 0.2 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    ok = y[:, None] + 0.8 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    bad = -y[:, None] + 1.0 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    return np.stack([good, ok, bad], axis=1), y


def test_greedy_prune_drops_bad_member() -> None:
    q, y = _members(80)
    out = greedy_prune(q, y, TAUS, max_steps=20)
    counts = np.asarray(out["counts"])
    assert counts[2] == 0.0  # the anti-correlated member is never added
    assert counts[0] > counts[1]


def test_pruned_weights_on_simplex() -> None:
    q, y = _members(81)
    counts = np.asarray(greedy_prune(q, y, TAUS, max_steps=10)["counts"])
    w = pruned_weights(counts)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert np.all(w >= 0)
    np.testing.assert_allclose(pruned_weights(np.zeros(3)), np.full(3, 1.0 / 3.0))


def test_pruning_gain_beats_equal_weight_with_bad_member() -> None:
    q, y = _members(82)
    out = pruning_gain(q, y, TAUS, max_steps=20)
    assert out["pruned_pinball"] < out["equal_pinball"]
    assert out["n_selected"] <= 3.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        greedy_prune(np.zeros((10, 2, 3)), np.zeros(9), TAUS)
    with pytest.raises(ValueError):
        pruned_weights(np.zeros((2, 2)))
