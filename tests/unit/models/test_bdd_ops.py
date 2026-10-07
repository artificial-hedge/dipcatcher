"""Tests for models/bdd_ops.py — sat_count must handle terminals and skips."""

from __future__ import annotations

import itertools


def test_sat_count_matches_truth() -> None:
    """The old sat_count indexed self.lo[0]/self.hi[0] on terminals and
    crashed; it also ignored variable-skip multiplicities."""
    from quant_fund.models.bdd_ops import _BDD, _eval

    order = ["a", "b", "c", "d"]
    bdd = _BDD()
    for form in [
        "(a and b) or (c and d)",
        "(a or b) and (not c or d)",
        "(a == b) and (c == d)",
    ]:
        root = bdd.build(form, order, {})
        truth = sum(
            1
            for vals in itertools.product([False, True], repeat=4)
            if _eval(form, dict(zip("abcd", vals, strict=True)))
        )
        assert bdd.sat_count(root, order) == truth


def test_sat_count_terminal_root() -> None:
    from quant_fund.models.bdd_ops import _BDD

    bdd = _BDD()
    assert bdd.sat_count(1, ["a", "b"]) == 4
    assert bdd.sat_count(0, ["a", "b"]) == 0
