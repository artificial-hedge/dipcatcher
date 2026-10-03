from quant_fund.models.d_critical import bench_d_critical
from quant_fund.models.derived_quot import bench_derived_quot
from quant_fund.models.intrinsic_be import bench_intrinsic_be
from quant_fund.models.perfect_obstruction import bench_perfect_obstruction
from quant_fund.models.shifted_tangent import bench_shifted_tangent
from quant_fund.models.virtual_pull import bench_virtual_pull


def test_shifted_tangent():
    assert bench_shifted_tangent()["synthetic_shifted_tangent"] == 1.0


def test_derived_quot():
    assert bench_derived_quot()["synthetic_derived_quot"] == 1.0


def test_virtual_pull():
    assert bench_virtual_pull()["synthetic_virtual_pull"] == 1.0


def test_intrinsic_be():
    assert bench_intrinsic_be()["synthetic_intrinsic_be"] == 1.0


def test_d_critical():
    assert bench_d_critical()["synthetic_d_critical"] == 1.0


def test_perfect_obstruction():
    assert bench_perfect_obstruction()["synthetic_perfect_obstruction"] == 1.0
