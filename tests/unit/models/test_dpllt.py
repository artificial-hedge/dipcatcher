"""Tests for the lazy DPLL(T) solver."""

from __future__ import annotations

import pytest

from quant_fund.models.dpllt import bench_dpllt, solve


def test_sat_chain_beyond_old_default_depth():
    """A 15-atom instance needs 15 decision levels — the old fixed
    depth=12 truncated the search and returned None (a silent UNSAT
    claim on an unexplored subtree)."""
    clauses = [frozenset({i, i + 1}) for i in range(1, 15)]
    out = solve(clauses, 15, lambda a: True)
    assert out is not None


def test_depth_exhaustion_raises_not_silent_unsat():
    clauses = [frozenset({i, i + 1}) for i in range(1, 15)]
    with pytest.raises(RuntimeError):
        solve(clauses, 15, lambda a: True, depth=3)


def test_unsat_still_none():
    out = solve([frozenset({1}), frozenset({-1})], 1, lambda a: True)
    assert out is None


def test_theory_prune():
    out = solve(
        [frozenset({1}), frozenset({2})],
        2,
        lambda a: not (a.get(1) and a.get(2)),
    )
    assert out is None


def test_bench():
    assert bench_dpllt()["synthetic_dpllt"] == 1.0
