"""Wave-311 VLSI-2 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.aig_rewrite import build_netlist, eval_aig, strash, sweep
from quant_fund.models.clock_tree import build_htree
from quant_fund.models.fm_partition import HyperGraph, _cut, fm_pass
from quant_fund.models.lee_router import route_net
from quant_fund.models.power_est import simulate, toggle_rates


def test_fm_balanced_and_nonworse() -> None:
    hg = HyperGraph(6, [[0, 1, 2], [3, 4, 5], [0, 3], [1, 4]])
    part = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    out = fm_pass(hg, part, tol=0.2)
    assert _cut(hg, out) <= _cut(hg, part)
    assert 1 <= int((out == 0).sum()) <= 4


def test_lee_shortest_on_open_grid() -> None:
    path = route_net(np.zeros((6, 6), dtype=int), [(0, 0), (5, 5)])
    assert path is not None
    assert int(path.sum()) == 11  # Manhattan path length 10 edges +1


def test_clock_tree_zero_skew() -> None:
    sinks = [(float(x), float(y)) for x in (0, 10) for y in (0, 10)]
    d = build_htree(sinks)
    assert len(d) == 4
    assert max(d.values()) - min(d.values()) < 1e-9


def test_aig_strash_preserves_function() -> None:
    ands, outs, n_in = build_netlist()
    ref = [eval_aig(ands, outs, n_in, x) for x in range(1 << n_in)]
    a2, o2 = strash(ands, outs, n_in)
    assert len(a2) < len(ands)
    a3, o3 = sweep(a2, o2, n_in)
    assert [eval_aig(a3, o3, n_in, x) for x in range(1 << n_in)] == ref


def test_power_xor_beats_and() -> None:
    n_in = 4
    xs = np.array([[0, 1, 0, 1], [1, 0, 1, 0]] * 8)
    states = simulate(n_in, [(0, 1), (n_in, 2)], xs)
    rates = toggle_rates(states)
    assert rates.shape[0] == n_in + 2
    assert (rates >= 0).all()
