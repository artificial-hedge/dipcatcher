from quant_fund.models.gromov_witten import bench_gromov_witten
from quant_fund.models.hilbert_scheme2 import bench_hilbert_scheme2
from quant_fund.models.kuranishi import bench_kuranishi
from quant_fund.models.m_bar_gn import bench_m_bar_gn
from quant_fund.models.quot_scheme import bench_quot_scheme
from quant_fund.models.stable_map import bench_stable_map


def test_kuranishi():
    assert bench_kuranishi()["synthetic_kuranishi"] == 1.0


def test_hilbert_scheme2():
    assert bench_hilbert_scheme2()["synthetic_hilbert_scheme2"] == 1.0


def test_quot_scheme():
    assert bench_quot_scheme()["synthetic_quot_scheme"] == 1.0


def test_m_bar_gn():
    assert bench_m_bar_gn()["synthetic_m_bar_gn"] == 1.0


def test_stable_map():
    assert bench_stable_map()["synthetic_stable_map"] == 1.0


def test_gromov_witten():
    assert bench_gromov_witten()["synthetic_gromov_witten"] == 1.0
