"""Tests for rrc_filter."""

from __future__ import annotations

import numpy as np

from quant_fund.models.rrc_filter import bench_rrc_filter


def test_rrc_nyquist_isi():
    from quant_fund.models.rrc_filter import rrc_taps

    sps = 8
    h = rrc_taps(sps, 8 * sps + 1, beta=0.35)
    hh = np.convolve(h, h)
    mid = len(hh) // 2
    assert abs(hh[mid + sps]) < 0.02
    assert abs(hh[mid + 2 * sps]) < 0.02


def test_bench_rrc():
    out = bench_rrc_filter(seed=20261231)
    assert out["synthetic_rrc_chain_ber"] < 0.05
    assert out["synthetic_rrc_isi_plus1"] < 0.02
    assert all(np.isfinite(v) for v in out.values())
