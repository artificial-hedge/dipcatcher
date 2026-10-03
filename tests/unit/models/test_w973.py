"""Wave-973 Banach-space-geometry canon tests."""

from __future__ import annotations

from quant_fund.models.banach_mazur import bench_banach_mazur
from quant_fund.models.djt_space import bench_djt_space
from quant_fund.models.gl_property import bench_gl_property
from quant_fund.models.kalton_loc import bench_kalton_loc
from quant_fund.models.schauder_basis import bench_schauder_basis
from quant_fund.models.type_cotype import bench_type_cotype


def test_banach_mazur():
    assert bench_banach_mazur()["synthetic_banach_mazur"] == 1.0


def test_type_cotype():
    assert bench_type_cotype()["synthetic_type_cotype"] == 1.0


def test_gl_property():
    assert bench_gl_property()["synthetic_gl_property"] == 1.0


def test_djt_space():
    assert bench_djt_space()["synthetic_djt_space"] == 1.0


def test_schauder_basis():
    assert bench_schauder_basis()["synthetic_schauder_basis"] == 1.0


def test_kalton_loc():
    assert bench_kalton_loc()["synthetic_kalton_loc"] == 1.0
