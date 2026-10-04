from quant_fund.models.bogdanov_takens import bench_bogdanov_takens
from quant_fund.models.homoclinic_bif import bench_homoclinic_bif
from quant_fund.models.hopf_bif import bench_hopf_bif
from quant_fund.models.neimark_sacker import bench_neimark_sacker
from quant_fund.models.period_doubling import bench_period_doubling
from quant_fund.models.saddle_node import bench_saddle_node


def test_saddle_node():
    assert bench_saddle_node()["synthetic_saddle_node"] == 1.0


def test_hopf_bif():
    assert bench_hopf_bif()["synthetic_hopf_bif"] == 1.0


def test_period_doubling():
    assert bench_period_doubling()["synthetic_period_doubling"] == 1.0


def test_neimark_sacker():
    assert bench_neimark_sacker()["synthetic_neimark_sacker"] == 1.0


def test_bogdanov_takens():
    assert bench_bogdanov_takens()["synthetic_bogdanov_takens"] == 1.0


def test_homoclinic_bif():
    assert bench_homoclinic_bif()["synthetic_homoclinic_bif"] == 1.0
