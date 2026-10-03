from quant_fund.models.derived_cohom import bench_derived_cohom
from quant_fund.models.derived_fiber2 import bench_derived_fiber2
from quant_fund.models.derived_intersection import (
    bench_derived_intersection,
)
from quant_fund.models.relative_trace import bench_relative_trace
from quant_fund.models.spectral_deformation2 import (
    bench_spectral_deformation2,
)
from quant_fund.models.virtual_class2 import bench_virtual_class2


def test_derived_cohom():
    assert bench_derived_cohom()["synthetic_derived_cohom"] == 1.0


def test_spectral_deformation2():
    assert bench_spectral_deformation2()["synthetic_spectral_deformation2"] == 1.0


def test_virtual_class2():
    assert bench_virtual_class2()["synthetic_virtual_class2"] == 1.0


def test_derived_intersection():
    assert bench_derived_intersection()["synthetic_derived_intersection"] == 1.0


def test_derived_fiber2():
    assert bench_derived_fiber2()["synthetic_derived_fiber2"] == 1.0


def test_relative_trace():
    assert bench_relative_trace()["synthetic_relative_trace"] == 1.0
