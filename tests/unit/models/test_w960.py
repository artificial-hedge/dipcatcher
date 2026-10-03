"""Wave-960 banach-algebra canon tests."""

from __future__ import annotations

from quant_fund.models.banach_algebra import bench_banach_algebra
from quant_fund.models.c_star_algebra import bench_c_star_algebra
from quant_fund.models.gelfand_transform import bench_gelfand_transform
from quant_fund.models.holomorphic_calculus import bench_holomorphic_calculus
from quant_fund.models.positive_functional import bench_positive_functional
from quant_fund.models.spectrum_algebra import bench_spectrum_algebra


def test_banach_algebra():
    assert bench_banach_algebra()["synthetic_banach_algebra"] == 1.0


def test_gelfand_transform():
    assert bench_gelfand_transform()["synthetic_gelfand_transform"] == 1.0


def test_c_star_algebra():
    assert bench_c_star_algebra()["synthetic_c_star_algebra"] == 1.0


def test_spectrum_algebra():
    assert bench_spectrum_algebra()["synthetic_spectrum_algebra"] == 1.0


def test_holomorphic_calculus():
    assert bench_holomorphic_calculus()["synthetic_holomorphic_calculus"] == 1.0


def test_positive_functional():
    assert bench_positive_functional()["synthetic_positive_functional"] == 1.0
