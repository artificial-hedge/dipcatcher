"""Probes for _sk_synth (spiking fixture)."""

import numpy as np
import pytest

from quant_fund.models._sk_synth import (
    latency_encode,
    lif_forward,
    poisson_encode,
    sk_data,
)


def test_sk_data_shapes_labels():
    X, y = sk_data(0, n=50, d=6)
    assert X.shape == (50, 6) and y.shape == (50,)
    assert set(np.unique(y)) <= {0, 1}


def test_sk_data_deterministic():
    a, b = sk_data(7, n=20), sk_data(7, n=20)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("kw", [{"n": 0}, {"d": 0}])
def test_sk_data_hostile(kw):
    with pytest.raises(ValueError):
        sk_data(0, **kw)


def test_poisson_encode_shape_rate_monotone():
    X = np.array([[10.0, -10.0]])  # high vs near-zero rate
    sp = poisson_encode(X, T=200, seed=0)
    assert sp.shape == (1, 2, 200)
    assert sp[0, 0].mean() > sp[0, 1].mean()
    assert set(np.unique(sp)) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "X,T",
    [
        (np.zeros(5), 10),  # 1-D input
        (np.zeros((0, 3)), 10),  # empty
        (np.zeros((2, 3)), 0),  # T=0 → empty spike tensor
        (np.zeros((2, 3)), -4),
    ],
)
def test_poisson_encode_hostile(X, T):
    with pytest.raises(ValueError):
        poisson_encode(X, T=T)


def test_lif_forward_decay():
    # single spike at t=0 decays exponentially: v_T = a^(T-1) * w
    sp = np.zeros((1, 1, 10))
    sp[0, 0, 0] = 1.0
    v = lif_forward(sp, np.array([2.0]), T=10)
    assert v[0] == pytest.approx(2.0 * 0.85**9)


@pytest.mark.parametrize(
    "sp,w,T",
    [
        (np.zeros((2, 3)), np.ones(3), 5),  # 2-D spikes
        (np.zeros((2, 3, 5)), np.ones(3), 0),  # T=0 → vacuous zeros
        (np.zeros((2, 3, 5)), np.ones(3), 6),  # T exceeds spike steps
        (np.zeros((2, 3, 5)), np.ones(2), 5),  # w dim mismatch
    ],
)
def test_lif_forward_hostile(sp, w, T):
    with pytest.raises(ValueError):
        lif_forward(sp, w, T=T)


def test_latency_encode_ordering():
    X = np.array([[5.0, -5.0]])  # high rate → early spike; low → late
    sp = latency_encode(X, T=20)
    assert sp.shape == (1, 2, 20)
    hi = np.argmax(sp[0, 0])
    lo = np.argmax(sp[0, 1])
    assert hi < lo
    assert sp.sum() == 2  # exactly one spike per input


def test_latency_encode_hostile():
    with pytest.raises(ValueError):
        latency_encode(np.zeros(4), T=10)
    with pytest.raises(ValueError):
        latency_encode(np.zeros((2, 2)), T=0)
