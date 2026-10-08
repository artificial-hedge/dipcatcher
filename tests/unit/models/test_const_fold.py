"""Tests for constant folding over straight-line SSA."""

from __future__ import annotations

import pytest

from quant_fund.models.const_fold import bench_const_fold, fold


def test_fold_add_mul() -> None:
    env = fold(
        [
            ("a", "const", 2),
            ("b", "const", 3),
            ("c", "add", "a", "b"),
            ("d", "mul", "c", "b"),
        ]
    )
    assert env["c"] == 5
    assert env["d"] == 15


def test_unknown_op_fails_closed() -> None:
    # An out-of-grammar opcode must not silently fold as multiply:
    # ("c","sub",4,2) used to produce 8 with no error.
    with pytest.raises(ValueError):
        fold([("a", "const", 4), ("b", "const", 2), ("c", "sub", "a", "b")])


def test_unresolved_operands_skipped() -> None:
    env = fold([("a", "add", "missing", 1), ("b", "const", 7)])
    assert "a" not in env
    assert env["b"] == 7


def test_bench_const_fold() -> None:
    assert bench_const_fold()["synthetic_fold_exact"] == 1.0
