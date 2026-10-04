"""Wave-992 scattering-theory canon tests."""

from __future__ import annotations

from quant_fund.models.limiting_absorption import bench_limiting_absorption
from quant_fund.models.radiation_cond import bench_radiation_cond
from quant_fund.models.resonances_thy import bench_resonances_thy
from quant_fund.models.scattering_matrix import bench_scattering_matrix
from quant_fund.models.trace_class_scatt import bench_trace_class_scatt
from quant_fund.models.wave_operators import bench_wave_operators


def test_wave_operators():
    assert bench_wave_operators()["synthetic_wave_operators"] == 1.0


def test_scattering_matrix():
    assert bench_scattering_matrix()["synthetic_scattering_matrix"] == 1.0


def test_limiting_absorption():
    assert bench_limiting_absorption()["synthetic_limiting_absorption"] == 1.0


def test_trace_class_scatt():
    assert bench_trace_class_scatt()["synthetic_trace_class_scatt"] == 1.0


def test_resonances_thy():
    assert bench_resonances_thy()["synthetic_resonances_thy"] == 1.0


def test_radiation_cond():
    assert bench_radiation_cond()["synthetic_radiation_cond"] == 1.0
