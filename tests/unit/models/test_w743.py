from quant_fund.models.baik_rmt import bench_baik_rmt
from quant_fund.models.borodin_olshanski import (
    bench_borodin_olshanski,
)
from quant_fund.models.bourgade_rmt import bench_bourgade_rmt
from quant_fund.models.chafai_rmt import bench_chafai_rmt
from quant_fund.models.cipolloni_erdos import bench_cipolloni_erdos
from quant_fund.models.tao_vu import bench_tao_vu


def test_baik_rmt():
    assert bench_baik_rmt()["synthetic_baik_rmt"] == 1.0


def test_tao_vu():
    assert bench_tao_vu()["synthetic_tao_vu"] == 1.0


def test_borodin_olshanski():
    assert bench_borodin_olshanski()["synthetic_borodin_olshanski"] == 1.0


def test_cipolloni_erdos():
    assert bench_cipolloni_erdos()["synthetic_cipolloni_erdos"] == 1.0


def test_bourgade_rmt():
    assert bench_bourgade_rmt()["synthetic_bourgade_rmt"] == 1.0


def test_chafai_rmt():
    assert bench_chafai_rmt()["synthetic_chafai_rmt"] == 1.0
