"""Adversarial probes for _td_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._td_synth import acc, make_data, train_mlp


def test_make_data_deterministic():
    a = make_data(0, n=64)
    b = make_data(0, n=64)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_make_data_hostile():
    with pytest.raises(ValueError):
        make_data(0, n=0)


def test_train_mlp_deterministic_and_loss_drops():
    torch = pytest.importorskip("torch")
    X, y, _, _ = make_data(0, n=128)
    net1, l1 = train_mlp(torch, X, y, iters=60, seed=0)
    net2, l2 = train_mlp(torch, X, y, iters=60, seed=0)
    assert l1 == l2  # seeded → identical trajectory
    assert len(l1) == 60
    assert l1[-1] < l1[0]  # actually learned something
    assert net1 is not net2


def test_train_mlp_isolates_global_rng():
    torch = pytest.importorskip("torch")
    X, y, _, _ = make_data(0, n=64)
    torch.manual_seed(123)
    pre = torch.random.get_rng_state()
    train_mlp(torch, X, y, iters=2, seed=7)
    post = torch.random.get_rng_state()
    assert torch.equal(pre, post)  # fork_rng must restore global state


@pytest.mark.parametrize(
    "iters,hidden,lr", [(0, 8, 0.01), (10, 0, 0.01), (10, 8, 0.0), (10, 8, -0.1), (10, 8, np.inf)]
)
def test_train_mlp_hostile(iters, hidden, lr):
    torch = pytest.importorskip("torch")
    X, y, _, _ = make_data(0, n=16)
    with pytest.raises(ValueError):
        train_mlp(torch, X, y, iters=iters, hidden=hidden, lr=lr)


def test_acc_range_and_hostile():
    torch = pytest.importorskip("torch")
    X, y, _, _ = make_data(0, n=64)
    net, _ = train_mlp(torch, X, y, iters=5, seed=0)
    a = acc(net, torch, X, y)
    assert 0.0 <= a <= 1.0
    with pytest.raises(ValueError):
        acc(net, torch, X[:0], y[:0])
    with pytest.raises(ValueError):
        acc(net, torch, X, y[: len(y) // 2])
