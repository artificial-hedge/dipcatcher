"""Wave-974 interpolation-theory canon tests."""

from __future__ import annotations

from quant_fund.models.complex_interp import bench_complex_interp
from quant_fund.models.lorentz_space import bench_lorentz_space
from quant_fund.models.marcinkiewicz_interp import bench_marcinkiewicz_interp
from quant_fund.models.peetre_kfunctor import bench_peetre_kfunctor
from quant_fund.models.real_interp_k import bench_real_interp_k
from quant_fund.models.reiteration_thm import bench_reiteration_thm


def test_real_interp_k():
    assert bench_real_interp_k()["synthetic_real_interp_k"] == 1.0


def test_complex_interp():
    assert bench_complex_interp()["synthetic_complex_interp"] == 1.0


def test_lorentz_space():
    assert bench_lorentz_space()["synthetic_lorentz_space"] == 1.0


def test_marcinkiewicz_interp():
    assert bench_marcinkiewicz_interp()["synthetic_marcinkiewicz_interp"] == 1.0


def test_peetre_kfunctor():
    assert bench_peetre_kfunctor()["synthetic_peetre_kfunctor"] == 1.0


def test_reiteration_thm():
    assert bench_reiteration_thm()["synthetic_reiteration_thm"] == 1.0
