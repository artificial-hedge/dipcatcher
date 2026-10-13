import numpy as np
import pytest

from quant_fund.decay.ic_matrix import (
    ic_factor_structure,
    ic_matrix,
    signal_clusters,
    signal_ic_correlation,
)

pytestmark = pytest.mark.synthetic


def _signals(seed: int, t_total: int = 400, n: int = 60):
    """Two signal families: A1/A2 share a component, B is independent."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((t_total, n))
    b_sig = rng.standard_normal((t_total, n))
    a1 = common + 0.3 * rng.standard_normal((t_total, n))
    a2 = common + 0.3 * rng.standard_normal((t_total, n))
    actual = common + b_sig + rng.standard_normal((t_total, n)) * 2.0
    return [a1, a2, b_sig], actual


def test_ic_matrix_shape() -> None:
    preds, actual = _signals(70)
    m = ic_matrix(preds, actual)
    assert m.shape == (400, 3)
    assert m[0, 0] > 0.1  # common component is predictable


def test_signal_correlation_clusters_families() -> None:
    preds, actual = _signals(71)
    corr = signal_ic_correlation(ic_matrix(preds, actual))
    assert corr.shape == (3, 3)
    assert corr[0, 1] > 0.3  # A1/A2 ICs co-move
    assert abs(corr[0, 2]) < abs(corr[0, 1])


def test_factor_structure_concentrated_for_shared_beta() -> None:
    preds, actual = _signals(72)
    out = ic_factor_structure(ic_matrix(preds, actual))
    assert 1.0 <= out["participation_ratio"] <= 3.0
    assert out["top_share"] > 0.4


def test_signal_clusters_groups_related_signals() -> None:
    preds, actual = _signals(73)
    labels = signal_clusters(ic_matrix(preds, actual), n_clusters=2)
    assert labels.shape == (3,)
    assert labels[0] == labels[1]  # A1, A2 together
    assert labels[2] != labels[0]


def test_validation() -> None:
    with pytest.raises(ValueError):
        ic_matrix([], np.zeros((5, 3)))
    with pytest.raises(ValueError):
        signal_ic_correlation(np.zeros((5, 1)))
    with pytest.raises(ValueError):
        signal_clusters(np.zeros((20, 2)), n_clusters=3)
