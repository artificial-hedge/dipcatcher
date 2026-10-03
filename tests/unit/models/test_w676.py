from quant_fund.models.center_hochschild import (
    bench_center_hochschild,
)
from quant_fund.models.en_algebra2 import bench_en_algebra2
from quant_fund.models.higher_brace2 import bench_higher_brace2
from quant_fund.models.koszul_operad2 import bench_koszul_operad2
from quant_fund.models.operad_lie import bench_operad_lie
from quant_fund.models.thom_transpose import bench_thom_transpose


def test_en_algebra2():
    assert bench_en_algebra2()["synthetic_en_algebra2"] == 1.0


def test_thom_transpose():
    assert bench_thom_transpose()["synthetic_thom_transpose"] == 1.0


def test_higher_brace2():
    assert bench_higher_brace2()["synthetic_higher_brace2"] == 1.0


def test_koszul_operad2():
    assert bench_koszul_operad2()["synthetic_koszul_operad2"] == 1.0


def test_operad_lie():
    assert bench_operad_lie()["synthetic_operad_lie"] == 1.0


def test_center_hochschild():
    assert bench_center_hochschild()["synthetic_center_hochschild"] == 1.0
