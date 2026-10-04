"""Wave-963 spectral-theory-2 canon tests."""

from __future__ import annotations

from quant_fund.models.atkinson_thm import bench_atkinson_thm
from quant_fund.models.browder_operator import bench_browder_operator
from quant_fund.models.essential_spectrum import bench_essential_spectrum
from quant_fund.models.fredholm_index import bench_fredholm_index
from quant_fund.models.riesz_schauder import bench_riesz_schauder
from quant_fund.models.weyl_theorem import bench_weyl_theorem


def test_fredholm_index():
    assert bench_fredholm_index()["synthetic_fredholm_index"] == 1.0


def test_weyl_theorem():
    assert bench_weyl_theorem()["synthetic_weyl_theorem"] == 1.0


def test_essential_spectrum():
    assert bench_essential_spectrum()["synthetic_essential_spectrum"] == 1.0


def test_browder_operator():
    assert bench_browder_operator()["synthetic_browder_operator"] == 1.0


def test_riesz_schauder():
    assert bench_riesz_schauder()["synthetic_riesz_schauder"] == 1.0


def test_atkinson_thm():
    assert bench_atkinson_thm()["synthetic_atkinson_thm"] == 1.0
