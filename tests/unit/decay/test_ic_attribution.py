import numpy as np
import pytest

from quant_fund.decay.ic_attribution import (
    group_ic_series,
    group_ic_summary,
    group_spread_permutation,
)

pytestmark = pytest.mark.synthetic


def _panel(seed: int, group_shift: float = 0.0):
    rng = np.random.default_rng(seed)
    t_total, n = 400, 60
    groups = np.repeat([0, 1, 2], 20)
    pred = rng.standard_normal((t_total, n))
    # group 2 gets an extra signal component when group_shift > 0
    actual = pred + rng.standard_normal((t_total, n)) * 3.0
    if group_shift:
        actual[:, groups == 2] += group_shift * pred[:, groups == 2]
    return pred, actual, groups


def test_group_ic_series_shape_and_keys() -> None:
    pred, actual, groups = _panel(40)
    series = group_ic_series(pred, actual, groups)
    assert set(series.keys()) == {0, 1, 2}
    for ic in series.values():
        assert ic.shape == (400,)


def test_group_summary_spread_detects_strong_group() -> None:
    pred, actual, groups = _panel(41, group_shift=1.0)
    series = group_ic_series(pred, actual, groups)
    out = group_ic_summary(series)
    means = out["mean_ic"]
    assert isinstance(means, np.ndarray)
    assert means[2] > means[0]
    assert means[2] > means[1]
    assert out["spread"] > 0.02


def test_group_spread_permutation_rejects_equal_groups() -> None:
    pred, actual, groups = _panel(42, group_shift=1.0)
    out = group_spread_permutation(pred, actual, groups, n_perm=300, seed=0)
    assert out["spread"] > 0.02
    assert out["p"] < 0.1


def test_group_spread_permutation_accepts_homogeneous_groups() -> None:
    pred, actual, groups = _panel(43, group_shift=0.0)
    out = group_spread_permutation(pred, actual, groups, n_perm=300, seed=0)
    assert out["p"] > 0.05


def test_small_groups_dropped() -> None:
    pred = np.random.default_rng(44).standard_normal((50, 6))
    actual = pred + np.random.default_rng(45).standard_normal((50, 6))
    groups = np.array([0, 0, 1, 1, 1, 1])
    series = group_ic_series(pred, actual, groups)
    assert set(series.keys()) == {1}
