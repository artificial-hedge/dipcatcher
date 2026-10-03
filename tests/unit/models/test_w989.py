"""Wave-989 microlocal-2 canon tests."""

from __future__ import annotations

from quant_fund.models.fbi_transform import bench_fbi_transform
from quant_fund.models.melrose_bdy import bench_melrose_bdy
from quant_fund.models.parametrix import bench_parametrix
from quant_fund.models.propagation_thm import bench_propagation_thm
from quant_fund.models.sg_calculus import bench_sg_calculus
from quant_fund.models.wave_eq_group import bench_wave_eq_group


def test_parametrix():
    assert bench_parametrix()["synthetic_parametrix"] == 1.0


def test_wave_eq_group():
    assert bench_wave_eq_group()["synthetic_wave_eq_group"] == 1.0


def test_propagation_thm():
    assert bench_propagation_thm()["synthetic_propagation_thm"] == 1.0


def test_melrose_bdy():
    assert bench_melrose_bdy()["synthetic_melrose_bdy"] == 1.0


def test_fbi_transform():
    assert bench_fbi_transform()["synthetic_fbi_transform"] == 1.0


def test_sg_calculus():
    assert bench_sg_calculus()["synthetic_sg_calculus"] == 1.0
