from quant_fund.models.freiman_thm import bench_freiman_thm
from quant_fund.models.gowers_norm import bench_gowers_norm
from quant_fund.models.green_tao import bench_green_tao
from quant_fund.models.plunnecke import bench_plunnecke
from quant_fund.models.roth_thm import bench_roth_thm
from quant_fund.models.szemeredi import bench_szemeredi


def test_freiman_thm():
    assert bench_freiman_thm()["synthetic_freiman_thm"] == 1.0


def test_szemeredi():
    assert bench_szemeredi()["synthetic_szemeredi"] == 1.0


def test_green_tao():
    assert bench_green_tao()["synthetic_green_tao"] == 1.0


def test_roth_thm():
    assert bench_roth_thm()["synthetic_roth_thm"] == 1.0


def test_gowers_norm():
    assert bench_gowers_norm()["synthetic_gowers_norm"] == 1.0


def test_plunnecke():
    assert bench_plunnecke()["synthetic_plunnecke"] == 1.0
