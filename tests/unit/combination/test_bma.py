import numpy as np
import pytest

from quant_fund.combination.bma import bma_combine, log_score_weights

pytestmark = pytest.mark.synthetic


def test_weights_concentrate_on_better_member() -> None:
    rng = np.random.default_rng(140)
    n = 300
    # member 0 scores ~ N(0.9, 0.05); member 1 ~ N(0.5, 0.05)
    scores = np.column_stack([rng.normal(0.9, 0.05, n), rng.normal(0.5, 0.05, n)])
    w = log_score_weights(scores, temperature=0.2)
    assert w.shape == (n, 2)
    assert w[-1, 0] > 0.99
    assert np.allclose(w.sum(axis=1), 1.0)


def test_forgetting_factor_reacts_to_switch() -> None:
    rng = np.random.default_rng(141)
    n = 100
    scores = np.empty((n, 2))
    scores[:50, 0] = rng.normal(0.9, 0.05, 50)
    scores[:50, 1] = rng.normal(0.5, 0.05, 50)
    scores[50:, 0] = rng.normal(0.5, 0.05, 50)
    scores[50:, 1] = rng.normal(0.9, 0.05, 50)
    w = log_score_weights(scores, temperature=0.2, forgetting=0.9)
    assert w[-1, 1] > w[-1, 0]


def test_no_forgetting_remembers_old_scores() -> None:
    rng = np.random.default_rng(142)
    n = 100
    scores = np.empty((n, 2))
    scores[:, 0] = rng.normal(0.9, 0.05, n)
    scores[:, 1] = rng.normal(0.5, 0.05, n)
    w = log_score_weights(scores, temperature=0.2, forgetting=1.0)
    # stable scores → weights stabilise
    assert abs(w[-1, 0] - w[-10, 0]) < 0.05


def test_bma_combine_weighted_average() -> None:
    preds = np.array([[1.0, 3.0], [2.0, 2.0]])
    weights = np.array([[0.25, 0.75], [0.5, 0.5]])
    out = bma_combine(preds, weights)
    np.testing.assert_allclose(out, [2.5, 2.0])


def test_bma_combine_validates_shapes() -> None:
    with pytest.raises(ValueError):
        bma_combine(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        log_score_weights(np.zeros(5), 1.0)
    with pytest.raises(ValueError):
        log_score_weights(np.zeros((5, 2)), 0.0)
