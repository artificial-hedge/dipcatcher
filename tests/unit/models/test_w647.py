from quant_fund.models.arkowitz_htpy import (
    bench_arkowitz_htpy,
)
from quant_fund.models.bochner_htpy import bench_bochner_htpy
from quant_fund.models.kahn_priddy import bench_kahn_priddy
from quant_fund.models.lin_htpy import bench_lin_htpy
from quant_fund.models.selick_htpy import bench_selick_htpy
from quant_fund.models.tits_building import (
    bench_tits_building,
)


def test_selick_htpy():
    assert bench_selick_htpy()["synthetic_selick_htpy"] == 1.0


def test_arkowitz_htpy():
    assert bench_arkowitz_htpy()["synthetic_arkowitz_htpy"] == 1.0


def test_lin_htpy():
    assert bench_lin_htpy()["synthetic_lin_htpy"] == 1.0


def test_kahn_priddy():
    assert bench_kahn_priddy()["synthetic_kahn_priddy"] == 1.0


def test_bochner_htpy():
    assert bench_bochner_htpy()["synthetic_bochner_htpy"] == 1.0


def test_tits_building():
    assert bench_tits_building()["synthetic_tits_building"] == 1.0
