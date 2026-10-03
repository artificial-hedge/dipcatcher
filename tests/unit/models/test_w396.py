from quant_fund.models.cm_points import bench_cm_points
from quant_fund.models.cyclotomic_field import bench_cyclotomic_field
from quant_fund.models.hensel_field import bench_hensel_field
from quant_fund.models.idele_class import bench_idele_class
from quant_fund.models.kronecker_weber import bench_kronecker_weber
from quant_fund.models.local_field import bench_local_field


def test_cyclotomic_field():
    assert bench_cyclotomic_field()["synthetic_cyclotomic_field"] == 1.0


def test_kronecker_weber():
    assert bench_kronecker_weber()["synthetic_kronecker_weber"] == 1.0


def test_local_field():
    assert bench_local_field()["synthetic_local_field"] == 1.0


def test_hensel_field():
    assert bench_hensel_field()["synthetic_hensel_field"] == 1.0


def test_cm_points():
    assert bench_cm_points()["synthetic_cm_points"] == 1.0


def test_idele_class():
    assert bench_idele_class()["synthetic_idele_class"] == 1.0
