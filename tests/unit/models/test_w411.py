from quant_fund.models.blow_up import bench_blow_up
from quant_fund.models.divisor_class import bench_divisor_class
from quant_fund.models.dualizing import bench_dualizing
from quant_fund.models.intersection_mult import bench_intersection_mult
from quant_fund.models.normalization import bench_normalization
from quant_fund.models.tangent_cone import bench_tangent_cone


def test_blow_up():
    assert bench_blow_up()["synthetic_blow_up"] == 1.0


def test_intersection_mult():
    assert bench_intersection_mult()["synthetic_intersection_mult"] == 1.0


def test_tangent_cone():
    assert bench_tangent_cone()["synthetic_tangent_cone"] == 1.0


def test_normalization():
    assert bench_normalization()["synthetic_normalization"] == 1.0


def test_divisor_class():
    assert bench_divisor_class()["synthetic_divisor_class"] == 1.0


def test_dualizing():
    assert bench_dualizing()["synthetic_dualizing"] == 1.0
