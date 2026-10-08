"""Tests for the Earley recognizer."""

from __future__ import annotations

from quant_fund.models.earley_parser import bench_earley_parser, earley_accepts


def test_accepts_balanced_parens():
    gram = [
        ("S", ("S", "S")),
        ("S", ("(", "S", ")")),
        ("S", ()),
    ]
    assert earley_accepts(gram, "S", list("(())()"))
    assert not earley_accepts(gram, "S", list("(()"))


def test_late_waiter_completes_against_epsilon_item():
    """Nullable nonterminal completes before a waiter on it is added.

    The only accepting derivation is S -> P Q -> Z X -> E X -> eps.
    Agenda order processes the dead-end Q -> E F branch first, popping
    (E -> .) before (Z -> .E) enters the chart: a snapshot complete pass
    never revisits completed items, so (Z -> .E) silently never advances
    and the recognizer wrongly rejects the empty word.
    """
    gram = [
        ("S", ("P", "Q")),
        ("P", ()),
        ("Q", ("Z", "X")),
        ("Q", ("E", "F")),
        ("E", ()),
        ("Z", ("E",)),
        ("F", ("f",)),
        ("X", ()),
    ]
    assert earley_accepts(gram, "S", [])


def test_rejects_unmatched_terminal():
    gram = [
        ("S", ("P", "Q")),
        ("P", ()),
        ("Q", ("Z", "X")),
        ("E", ()),
        ("Z", ("E",)),
        ("X", ()),
    ]
    assert not earley_accepts(gram, "S", ["q"])


def test_bench():
    out = bench_earley_parser()
    assert out["synthetic_parens_correct"] == 1.0
    assert out["synthetic_rejects_invalid"] == 1.0
