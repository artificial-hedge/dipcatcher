from quant_fund.models.aubin_thm import bench_aubin_thm
from quant_fund.models.kazdan_warner import bench_kazdan_warner
from quant_fund.models.nirenberg_problem import (
    bench_nirenberg_problem,
)
from quant_fund.models.prescribed_curvature import (
    bench_prescribed_curvature,
)
from quant_fund.models.trudinger_thm import bench_trudinger_thm
from quant_fund.models.yamabe_problem import bench_yamabe_problem


def test_yamabe_problem():
    assert bench_yamabe_problem()["synthetic_yamabe_problem"] == 1.0


def test_prescribed_curvature():
    assert bench_prescribed_curvature()["synthetic_prescribed_curvature"] == 1.0


def test_nirenberg_problem():
    assert bench_nirenberg_problem()["synthetic_nirenberg_problem"] == 1.0


def test_kazdan_warner():
    assert bench_kazdan_warner()["synthetic_kazdan_warner"] == 1.0


def test_aubin_thm():
    assert bench_aubin_thm()["synthetic_aubin_thm"] == 1.0


def test_trudinger_thm():
    assert bench_trudinger_thm()["synthetic_trudinger_thm"] == 1.0
