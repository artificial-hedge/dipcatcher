from quant_fund.models.hauwas_nori import bench_hauwas_nori
from quant_fund.models.jogiad_motive import (
    bench_jogiad_motive,
)
from quant_fund.models.motivic_pipe import bench_motivic_pipe
from quant_fund.models.roald_suslin import (
    bench_roald_suslin,
)
from quant_fund.models.thom_mgl2 import bench_thom_mgl2
from quant_fund.models.voev_suslin import bench_voev_suslin


def test_roald_suslin():
    assert bench_roald_suslin()["synthetic_roald_suslin"] == 1.0


def test_jogiad_motive():
    assert bench_jogiad_motive()["synthetic_jogiad_motive"] == 1.0


def test_hauwas_nori():
    assert bench_hauwas_nori()["synthetic_hauwas_nori"] == 1.0


def test_motivic_pipe():
    assert bench_motivic_pipe()["synthetic_motivic_pipe"] == 1.0


def test_thom_mgl2():
    assert bench_thom_mgl2()["synthetic_thom_mgl2"] == 1.0


def test_voev_suslin():
    assert bench_voev_suslin()["synthetic_voev_suslin"] == 1.0
