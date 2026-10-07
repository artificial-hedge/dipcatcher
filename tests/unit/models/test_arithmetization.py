"""Tests for models/arithmetization.py — Gödel beta sequence coding."""

from __future__ import annotations

import pytest

from quant_fund.models.arithmetization import (
    _bench_arithmetization,
    beta,
    code_seq,
)


def test_roundtrip_basic() -> None:
    seq = [3, 1, 4, 1, 5, 9]
    a, b = code_seq(seq)
    assert [beta(a, b, i) for i in range(len(seq))] == seq


def test_negative_elements_raise() -> None:
    """Negative elements silently produced non-round-tripping codes —
    must fail closed."""
    with pytest.raises(ValueError):
        code_seq([-3, 1, 4])


def test_empty_seq_raises() -> None:
    with pytest.raises(ValueError):
        code_seq([])


def test_zero_elements_ok() -> None:
    a, b = code_seq([0, 2, 0])
    assert [beta(a, b, i) for i in range(3)] == [0, 2, 0]


def test_bench() -> None:
    assert _bench_arithmetization() == 1.0
