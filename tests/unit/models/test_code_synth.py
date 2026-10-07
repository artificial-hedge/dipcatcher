"""Unit tests for quant_fund.models._code_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._code_synth import (
    bpsk_awgn,
    bsc,
    burst_channel,
    msg_bits,
)


def test_msg_bits_deterministic_binary() -> None:
    a = msg_bits(3, 32)
    b = msg_bits(3, 32)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0, 1}


def test_bsc_deterministic_and_extremes() -> None:
    bits = np.zeros(50, dtype=int)
    a = bsc(bits, 0.3, 4)
    b = bsc(bits, 0.3, 4)
    assert np.array_equal(a, b)
    assert np.array_equal(bsc(bits, 0.0, 4), bits)
    assert np.array_equal(bsc(bits, 1.0, 4), bits ^ 1)


def test_bsc_rejects_bad_probability() -> None:
    with pytest.raises(ValueError, match="crossover"):
        bsc(np.zeros(8, dtype=int), -0.1, 0)
    with pytest.raises(ValueError, match="crossover"):
        bsc(np.zeros(8, dtype=int), 1.5, 0)


def test_burst_channel_flips_in_bounds_only() -> None:
    bits = np.zeros(10, dtype=int)
    out = burst_channel(bits, 3, 4)
    assert np.array_equal(out, np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))
    with pytest.raises(ValueError, match="burst"):
        burst_channel(bits, -2, 3)
    with pytest.raises(ValueError, match="burst"):
        burst_channel(bits, 8, 5)
    with pytest.raises(ValueError, match="burst"):
        burst_channel(bits, 2, -1)


def test_bpsk_sign_convention() -> None:
    bits = np.array([0, 1])
    out = bpsk_awgn(bits, 0.0, 0)
    # 0 -> +1, 1 -> -1 with zero noise
    assert np.allclose(out, [1.0, -1.0])
