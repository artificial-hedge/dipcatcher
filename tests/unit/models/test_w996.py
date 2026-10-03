"""Wave-996 Riemann-Hilbert canon tests."""

from __future__ import annotations

from quant_fund.models.dbar_method import bench_dbar_method
from quant_fund.models.deift_zhou import bench_deift_zhou
from quant_fund.models.fokas_unified import bench_fokas_unified
from quant_fund.models.isomonodromy import bench_isomonodromy
from quant_fund.models.orthogonal_poly_rh import bench_orthogonal_poly_rh
from quant_fund.models.small_norm_rh import bench_small_norm_rh


def test_dbar_method():
    assert bench_dbar_method()["synthetic_dbar_method"] == 1.0


def test_orthogonal_poly_rh():
    assert bench_orthogonal_poly_rh()["synthetic_orthogonal_poly_rh"] == 1.0


def test_isomonodromy():
    assert bench_isomonodromy()["synthetic_isomonodromy"] == 1.0


def test_fokas_unified():
    assert bench_fokas_unified()["synthetic_fokas_unified"] == 1.0


def test_deift_zhou():
    assert bench_deift_zhou()["synthetic_deift_zhou"] == 1.0


def test_small_norm_rh():
    assert bench_small_norm_rh()["synthetic_small_norm_rh"] == 1.0
