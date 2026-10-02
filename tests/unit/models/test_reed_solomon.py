"""Tests for reed_solomon."""

from __future__ import annotations

import numpy as np

from quant_fund.models.reed_solomon import bench_reed_solomon


def test_rs_clean():
    from quant_fund.models.reed_solomon import rs_decode, rs_encode

    msg = [1, 2, 3, 4, 5]
    code = rs_encode(msg, 4)
    dec, ne = rs_decode(code, 4)
    assert dec == msg
    assert ne == 0


def test_rs_corrects_t():
    from quant_fund.models.reed_solomon import rs_decode, rs_encode

    msg = list(range(50))
    code = rs_encode(msg, 8)
    code[3] ^= 0xAB
    code[20] ^= 0x55
    dec, ne = rs_decode(code, 8)
    assert dec == msg
    assert ne == 2


def test_bench_rs():
    out = bench_reed_solomon(seed=20261231)
    assert out["synthetic_rs_clean_ber"] == 0.0
    assert out["synthetic_rs_t5_fixed"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
