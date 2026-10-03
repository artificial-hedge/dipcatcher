"""Tests for gardner."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gardner import bench_gardner


def test_gardner_aligned():
    from quant_fund.models.gardner import gardner_recover
    from quant_fund.models.rrc_filter import matched_filter, pulse_shape, rrc_taps

    rng = np.random.default_rng(4)
    sps = 8
    bits = (2 * rng.integers(0, 2, 120) - 1).astype(np.float64)
    h = rrc_taps(sps, 8 * sps + 1, beta=0.35)
    mf = matched_filter(pulse_shape(bits, sps, h), h)
    dec, _ = gardner_recover(mf[: 100 * sps], sps=sps)
    assert np.mean(dec[20:] != bits[20 : len(dec)]) < 0.05


def test_bench_gardner():
    out = bench_gardner(seed=20261231)
    assert out["synthetic_gardner_ber"] < 0.05
    assert out["synthetic_gardner_gain"] >= 0.0
    assert all(np.isfinite(v) for v in out.values())
