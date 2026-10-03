from quant_fund.models.logarithmic_red import (
    bench_logarithmic_red,
)
from quant_fund.models.matrix_geom import bench_matrix_geom
from quant_fund.models.neuts_map import bench_neuts_map
from quant_fund.models.phase_type import bench_phase_type
from quant_fund.models.quasi_birth import bench_quasi_birth
from quant_fund.models.ramaswami import bench_ramaswami


def test_neuts_map():
    assert bench_neuts_map()["synthetic_neuts_map"] == 1.0


def test_phase_type():
    assert bench_phase_type()["synthetic_phase_type"] == 1.0


def test_matrix_geom():
    assert bench_matrix_geom()["synthetic_matrix_geom"] == 1.0


def test_quasi_birth():
    assert bench_quasi_birth()["synthetic_quasi_birth"] == 1.0


def test_ramaswami():
    assert bench_ramaswami()["synthetic_ramaswami"] == 1.0


def test_logarithmic_red():
    assert bench_logarithmic_red()["synthetic_logarithmic_red"] == 1.0
