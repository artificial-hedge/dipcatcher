from quant_fund.models.cyclotomic_poly import bench_cyclotomic_poly
from quant_fund.models.finite_field import bench_finite_field
from quant_fund.models.galois_corresp import bench_galois_corresp
from quant_fund.models.normality_check import bench_normality_check
from quant_fund.models.primitive_elem import bench_primitive_elem
from quant_fund.models.separable_check import bench_separable_check


def test_finite_field():
    assert bench_finite_field()["synthetic_finite_field"] == 1.0


def test_galois_corresp():
    assert bench_galois_corresp()["synthetic_galois_corresp"] == 1.0


def test_normality_check():
    assert bench_normality_check()["synthetic_normality_check"] == 1.0


def test_separable_check():
    assert bench_separable_check()["synthetic_separable_check"] == 1.0


def test_cyclotomic_poly():
    assert bench_cyclotomic_poly()["synthetic_cyclotomic_poly"] == 1.0


def test_primitive_elem():
    assert bench_primitive_elem()["synthetic_primitive_elem"] == 1.0
