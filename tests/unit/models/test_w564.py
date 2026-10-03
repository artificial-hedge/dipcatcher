from quant_fund.models.asymptotic_cone import bench_asymptotic_cone
from quant_fund.models.baumslag_solitar import bench_baumslag_solitar
from quant_fund.models.gromov_hyperbolic import (
    bench_gromov_hyperbolic,
)
from quant_fund.models.quasi_isometry import bench_quasi_isometry
from quant_fund.models.thin_triangle import bench_thin_triangle
from quant_fund.models.word_problem import bench_word_problem


def test_gromov_hyperbolic():
    assert bench_gromov_hyperbolic()["synthetic_gromov_hyperbolic"] == 1.0


def test_quasi_isometry():
    assert bench_quasi_isometry()["synthetic_quasi_isometry"] == 1.0


def test_thin_triangle():
    assert bench_thin_triangle()["synthetic_thin_triangle"] == 1.0


def test_word_problem():
    assert bench_word_problem()["synthetic_word_problem"] == 1.0


def test_baumslag_solitar():
    assert bench_baumslag_solitar()["synthetic_baumslag_solitar"] == 1.0


def test_asymptotic_cone():
    assert bench_asymptotic_cone()["synthetic_asymptotic_cone"] == 1.0
