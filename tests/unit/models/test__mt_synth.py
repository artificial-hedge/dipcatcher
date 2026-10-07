"""Probes for _mt_synth.mt_data."""

import numpy as np
import pytest

from quant_fund.models._mt_synth import mt_data


def test_shapes_and_labels():
    X, y1, y2 = mt_data(0, n=64, d=8)
    assert X.shape == (64, 8)
    assert y1.shape == y2.shape == (64,)
    assert set(np.unique(y1)) <= {0.0, 1.0}
    assert set(np.unique(y2)) <= {0.0, 1.0}


def test_deterministic():
    a = mt_data(42, n=32)
    b = mt_data(42, n=32)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kw", [{"n": 0}, {"n": -5}, {"d": 0}, {"d": -1}])
def test_hostile_params_raise(kw):
    with pytest.raises(ValueError):
        mt_data(0, **kw)


def test_tasks_not_identical():
    _, y1, y2 = mt_data(0, n=200)
    assert not np.array_equal(y1, y2)
