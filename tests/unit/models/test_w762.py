from quant_fund.models.cadlag_space import bench_cadlag_space
from quant_fund.models.doob_meyer import bench_doob_meyer
from quant_fund.models.martin_boundary import bench_martin_boundary
from quant_fund.models.prohorov_thm2 import bench_prohorov_thm2
from quant_fund.models.skohorod_metric import (
    bench_skohorod_metric,
)
from quant_fund.models.tightness_check import bench_tightness_check


def test_martin_boundary():
    assert bench_martin_boundary()["synthetic_martin_boundary"] == 1.0


def test_doob_meyer():
    assert bench_doob_meyer()["synthetic_doob_meyer"] == 1.0


def test_cadlag_space():
    assert bench_cadlag_space()["synthetic_cadlag_space"] == 1.0


def test_skohorod_metric():
    assert bench_skohorod_metric()["synthetic_skohorod_metric"] == 1.0


def test_prohorov_thm2():
    assert bench_prohorov_thm2()["synthetic_prohorov_thm2"] == 1.0


def test_tightness_check():
    assert bench_tightness_check()["synthetic_tightness_check"] == 1.0
