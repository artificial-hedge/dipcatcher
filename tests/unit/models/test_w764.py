from quant_fund.models.donsker_class import bench_donsker_class
from quant_fund.models.donsker_thm import bench_donsker_thm
from quant_fund.models.dz_invariance import bench_dz_invariance
from quant_fund.models.empirical_process import (
    bench_empirical_process,
)
from quant_fund.models.osj_metric import bench_osj_metric
from quant_fund.models.wiener_measure import bench_wiener_measure


def test_wiener_measure():
    assert bench_wiener_measure()["synthetic_wiener_measure"] == 1.0


def test_dz_invariance():
    assert bench_dz_invariance()["synthetic_dz_invariance"] == 1.0


def test_donsker_thm():
    assert bench_donsker_thm()["synthetic_donsker_thm"] == 1.0


def test_empirical_process():
    assert bench_empirical_process()["synthetic_empirical_process"] == 1.0


def test_donsker_class():
    assert bench_donsker_class()["synthetic_donsker_class"] == 1.0


def test_osj_metric():
    assert bench_osj_metric()["synthetic_osj_metric"] == 1.0
