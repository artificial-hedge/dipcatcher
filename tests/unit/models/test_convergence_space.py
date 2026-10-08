"""Tests for sequence convergence on finite topologies."""

from __future__ import annotations

from quant_fund.models.convergence_space import (
    all_limits,
    bench_convergence_space,
    is_hausdorff,
    seq_converges,
)

_U = frozenset({0, 1})
_DISC = frozenset({frozenset(), frozenset({0}), frozenset({1}), _U})
_IND = frozenset({frozenset(), _U})


def test_tail_semantics_last_element() -> None:
    # "Eventually in U" means a tail lies inside U — for a finite list that
    # is the last element. [0,1] converges to 1, not to 0, in the discrete
    # topology; the whole-list check wrongly rejected it.
    assert seq_converges([0, 1], 1, _DISC)
    assert not seq_converges([0, 1], 0, _DISC)
    assert all_limits([0, 1], _U, _DISC) == frozenset({1})


def test_constant_sequence() -> None:
    assert all_limits([0, 0], _U, _DISC) == frozenset({0})
    # indiscrete: only nbhd is the whole space -> converges to everything
    assert all_limits([0, 1], _U, _IND) == frozenset({0, 1})


def test_hausdorff() -> None:
    assert is_hausdorff(_U, _DISC)
    assert not is_hausdorff(_U, _IND)


def test_bench_convergence_space() -> None:
    assert bench_convergence_space()["synthetic_convergence_space"] == 1.0
