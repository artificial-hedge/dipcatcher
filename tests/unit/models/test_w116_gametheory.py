"""Tests for wave-116 game-theory canon: cfr, lemke_howson,
replicator, wardrop, vcg, nash_bargain."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cfr import (
    _kuhn_util,
    bench_cfr,
    cfr_kuhn,
    kuhn_exploitability,
)
from quant_fund.models.lemke_howson import bench_lemke_howson, lemke_howson
from quant_fund.models.nash_bargain import (
    bench_nash_bargain,
    nash_bargain_linear,
)
from quant_fund.models.replicator import bench_replicator, replicator_run
from quant_fund.models.vcg import bench_vcg, vcg_auction
from quant_fund.models.wardrop import bench_wardrop, wardrop_solve


def test_kuhn_util_terminals():
    assert _kuhn_util("pp", 2, 0, 1) == 1.0
    assert _kuhn_util("bb", 0, 2, 1) == -2.0
    assert _kuhn_util("bp", 0, 2, 1) == 1.0
    assert _kuhn_util("pbp", 0, 2, 2) == 1.0


def test_cfr_converges():
    rng = np.random.default_rng(0)
    avg = cfr_kuhn(5_000, rng)
    expl = kuhn_exploitability(avg)
    assert expl < 0.25  # well below the ~0.6 uniform-policy level


def test_lemke_howson_pure():
    # prisoner's-dilemma-ish unique pure eq: both defect
    A = np.array([[2.0, 0], [3, 1]])
    B = np.array([[2.0, 3], [0, 1]])
    x, y = lemke_howson(A, B)
    Ay = A @ y
    assert Ay[x > 1e-9].min() >= Ay.max() - 1e-7


def test_replicator_ess():
    v, g = 2.0, 4.0
    A = np.array([[(v - g) / 2, v], [0.0, v / 2]])
    x = replicator_run(np.array([0.9, 0.1]), A, 0.05, 5_000)
    assert abs(x[0] - 0.5) < 0.05


def test_wardrop_pigou():
    free = np.array([1.0, 0.0])
    cap = np.array([1.0, 1.0])
    f = wardrop_solve([[0], [1]], 1.0, free, cap, iters=2000, b=np.array([0.0, 1.0]), p=1.0)
    assert f[1] > 0.9


def test_vcg_second_price():
    alloc, pay = vcg_auction(np.array([5.0, 3, 1]), 1)
    assert alloc[0] == 1 and pay[0] == pytest.approx(3.0)


def test_nash_bargain_symmetric():
    u = nash_bargain_linear(1.0, 1.0, np.array([0.0, 0.0]))
    assert np.linalg.norm(u - 0.5) < 1e-9


@pytest.mark.parametrize(
    "fn",
    [bench_cfr, bench_lemke_howson, bench_replicator, bench_wardrop, bench_vcg, bench_nash_bargain],
    ids=lambda f: f.__name__,
)
def test_w116_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
