from quant_fund.models.dw_ldp import bench_dw_ldp
from quant_fund.models.freidlin_wentzell import (
    bench_freidlin_wentzell,
)
from quant_fund.models.mogulskii_thm import (
    bench_mogulskii_thm,
)
from quant_fund.models.sanov_thm import bench_sanov_thm
from quant_fund.models.schider_thm import bench_schider_thm
from quant_fund.models.varadhan_ldp import bench_varadhan_ldp


def test_varadhan_ldp():
    assert bench_varadhan_ldp()["synthetic_varadhan_ldp"] == 1.0


def test_freidlin_wentzell():
    assert bench_freidlin_wentzell()["synthetic_freidlin_wentzell"] == 1.0


def test_dw_ldp():
    assert bench_dw_ldp()["synthetic_dw_ldp"] == 1.0


def test_sanov_thm():
    assert bench_sanov_thm()["synthetic_sanov_thm"] == 1.0


def test_mogulskii_thm():
    assert bench_mogulskii_thm()["synthetic_mogulskii_thm"] == 1.0


def test_schider_thm():
    assert bench_schider_thm()["synthetic_schider_thm"] == 1.0
