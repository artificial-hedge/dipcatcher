"""Tests for gf256."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gf256 import bench_gf256


def test_gf_mul_inverse():
    from quant_fund.models.gf256 import gf_inv, gf_mul

    for a in (1, 2, 3, 7, 200, 255):
        assert gf_mul(a, gf_inv(a)) == 1


def test_gf_div_roundtrip():
    from quant_fund.models.gf256 import gf_div, gf_mul

    assert gf_div(gf_mul(37, 19), 19) == 37


def test_bench_gf256():
    out = bench_gf256(seed=20261231)
    assert out["synthetic_gf_axiom_violations"] == 0.0
    assert out["synthetic_gf_genpoly_roots"] == 0.0
    assert all(np.isfinite(v) for v in out.values())
