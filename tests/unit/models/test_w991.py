"""Wave-991 homogenization canon tests."""

from __future__ import annotations

from quant_fund.models.bloch_decomp import bench_bloch_decomp
from quant_fund.models.gamma_convergence import bench_gamma_convergence
from quant_fund.models.h_convergence import bench_h_convergence
from quant_fund.models.homogenization import bench_homogenization
from quant_fund.models.mosco_conv import bench_mosco_conv
from quant_fund.models.two_scale_conv import bench_two_scale_conv


def test_homogenization():
    assert bench_homogenization()["synthetic_homogenization"] == 1.0


def test_two_scale_conv():
    assert bench_two_scale_conv()["synthetic_two_scale_conv"] == 1.0


def test_gamma_convergence():
    assert bench_gamma_convergence()["synthetic_gamma_convergence"] == 1.0


def test_mosco_conv():
    assert bench_mosco_conv()["synthetic_mosco_conv"] == 1.0


def test_bloch_decomp():
    assert bench_bloch_decomp()["synthetic_bloch_decomp"] == 1.0


def test_h_convergence():
    assert bench_h_convergence()["synthetic_h_convergence"] == 1.0
