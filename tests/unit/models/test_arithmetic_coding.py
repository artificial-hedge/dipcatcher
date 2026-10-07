"""Tests for models/arithmetic_coding.py — integer range coder."""

from __future__ import annotations

import pytest

from quant_fund.models.arithmetic_coding import (
    TOP,
    bench_arithmetic_coding,
    decode,
    encode,
)


def test_roundtrip_small() -> None:
    freq = {0: 5, 1: 3, 2: 1}
    msg = [0, 1, 2, 0, 0, 1, 2, 2]
    bits, _ = encode(msg, freq)
    assert decode(bits, len(msg), freq) == msg


def test_oversized_frequency_total_raises() -> None:
    """freq total > 2**32 makes the renorm loop diverge (hang) — must
    fail closed instead."""
    freq = {0: 2 * TOP, 1: 1}
    with pytest.raises(ValueError):
        encode([1, 1], freq)
    with pytest.raises(ValueError):
        decode([1] * 10, 4, freq)


def test_boundary_total_ok() -> None:
    freq = {0: TOP - 2, 1: 2}
    bits, _ = encode([0, 1, 0], freq)
    assert decode(bits, 3, freq) == [0, 1, 0]


def test_bench_arithmetic_coding() -> None:
    out = bench_arithmetic_coding()
    assert out["synthetic_roundtrip"] == 1.0
