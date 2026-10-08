"""Tests for closed_graph — finite-dim closed-graph checks."""

from __future__ import annotations

import numpy as np

from quant_fund.models.closed_graph import bench_closed_graph, boundedness, graph_is_closed


def test_membership_check_discriminates():
    # the membership test must reject a limit that is off the graph —
    # the old tautological version compared a @ x to itself and could not.
    from quant_fund.models.closed_graph import _on_graph

    a = np.array([[2.0, 0.0], [0.0, 0.5]])
    x = np.array([1.0, -1.0])
    assert _on_graph(a, x, a @ x, 1e-9)
    assert not _on_graph(a, x, a @ x + np.array([0.0, 1e-3]), 1e-9)


def test_graph_is_closed_linear():
    a = np.random.default_rng(0).normal(size=(4, 4))
    assert graph_is_closed(a, np.random.default_rng(1))


def test_boundedness():
    assert boundedness(np.diag([3.0, 1.0])) == 3.0


def test_bench():
    assert bench_closed_graph()["synthetic_closed_graph"] == 1.0
