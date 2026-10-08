"""Tests for dfa_minimize — Hopcroft minimization + table-filling oracle."""

from __future__ import annotations


def test_distinguishable_pairs_dead_end_vs_live():
    """A missing transition is a rejecting dead end, not a jump to state 0.

    State 1 accepts the string "0" (reaches accept state 0); state 2 has
    no "0" transition and accepts nothing — the pair is distinguishable.
    """
    from quant_fund.models.dfa_minimize import distinguishable_pairs

    trans = [{"0": 0}, {"0": 0}, {}]
    marked = distinguishable_pairs(3, ["0"], trans, {0})
    assert frozenset({1, 2}) in marked


def test_distinguishable_pairs_dead_equivalent():
    """A dead end and a state that can never reach accept are equivalent."""
    from quant_fund.models.dfa_minimize import distinguishable_pairs

    trans = [{"0": 0}, {"0": 1}, {}]
    marked = distinguishable_pairs(3, ["0"], trans, {0})
    assert frozenset({1, 2}) not in marked


def test_bench_dfa_minimize():
    from quant_fund.models.dfa_minimize import bench_dfa_minimize

    out = bench_dfa_minimize()
    assert out["synthetic_partition_ok"] == 1.0
    assert out["synthetic_lang_equiv"] == 1.0
    assert out["synthetic_shrinks"] == 1.0
