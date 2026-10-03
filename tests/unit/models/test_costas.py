"""Tests for costas."""

from __future__ import annotations

import numpy as np

from quant_fund.models.costas import bench_costas


def test_costas_static_offset():
    from quant_fund.models.costas import bpsk_symbols, costas_loop

    bits = bpsk_symbols(400, seed=2)
    rx = bits * np.exp(1j * 0.5)
    dec, phases, _ = costas_loop(rx)
    st = 100
    assert np.mean(dec[st:] != bits[st:]) < 0.05


def test_bench_costas():
    out = bench_costas(seed=20261231)
    assert out["synthetic_costas_static_ber"] < 0.05
    assert out["synthetic_costas_freq_ber"] < 0.05
    assert all(np.isfinite(v) for v in out.values())
