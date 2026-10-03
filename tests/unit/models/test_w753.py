from quant_fund.models.aizenman_irf import bench_aizenman_irf
from quant_fund.models.cardy_on import bench_cardy_on
from quant_fund.models.fernandez_frohlich import (
    bench_fernandez_frohlich,
)
from quant_fund.models.fradkin_sokal import bench_fradkin_sokal
from quant_fund.models.nienhuis_on import bench_nienhuis_on
from quant_fund.models.pelissetto_vicari import (
    bench_pelissetto_vicari,
)


def test_fernandez_frohlich():
    assert bench_fernandez_frohlich()["synthetic_fernandez_frohlich"] == 1.0


def test_aizenman_irf():
    assert bench_aizenman_irf()["synthetic_aizenman_irf"] == 1.0


def test_fradkin_sokal():
    assert bench_fradkin_sokal()["synthetic_fradkin_sokal"] == 1.0


def test_nienhuis_on():
    assert bench_nienhuis_on()["synthetic_nienhuis_on"] == 1.0


def test_cardy_on():
    assert bench_cardy_on()["synthetic_cardy_on"] == 1.0


def test_pelissetto_vicari():
    assert bench_pelissetto_vicari()["synthetic_pelissetto_vicari"] == 1.0
