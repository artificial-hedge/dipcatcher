from quant_fund.models.braces_e5 import bench_braces_e5
from quant_fund.models.delooping3 import bench_delooping3
from quant_fund.models.higher_algebra9 import (
    bench_higher_algebra9,
)
from quant_fund.models.koszul_duality3 import (
    bench_koszul_duality3,
)
from quant_fund.models.operad_infty5 import bench_operad_infty5
from quant_fund.models.operad_swiss4 import bench_operad_swiss4


def test_higher_algebra9():
    assert bench_higher_algebra9()["synthetic_higher_algebra9"] == 1.0


def test_operad_infty5():
    assert bench_operad_infty5()["synthetic_operad_infty5"] == 1.0


def test_operad_swiss4():
    assert bench_operad_swiss4()["synthetic_operad_swiss4"] == 1.0


def test_koszul_duality3():
    assert bench_koszul_duality3()["synthetic_koszul_duality3"] == 1.0


def test_braces_e5():
    assert bench_braces_e5()["synthetic_braces_e5"] == 1.0


def test_delooping3():
    assert bench_delooping3()["synthetic_delooping3"] == 1.0
