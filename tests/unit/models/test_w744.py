from quant_fund.models.amir_corwin import bench_amir_corwin
from quant_fund.models.borodin_corwin import bench_borodin_corwin
from quant_fund.models.calabrese_kpz import bench_calabrese_kpz
from quant_fund.models.corwin_kpz import bench_corwin_kpz
from quant_fund.models.kardar_parisi import bench_kardar_parisi
from quant_fund.models.quastel_spohn import bench_quastel_spohn


def test_kardar_parisi():
    assert bench_kardar_parisi()["synthetic_kardar_parisi"] == 1.0


def test_corwin_kpz():
    assert bench_corwin_kpz()["synthetic_corwin_kpz"] == 1.0


def test_quastel_spohn():
    assert bench_quastel_spohn()["synthetic_quastel_spohn"] == 1.0


def test_borodin_corwin():
    assert bench_borodin_corwin()["synthetic_borodin_corwin"] == 1.0


def test_amir_corwin():
    assert bench_amir_corwin()["synthetic_amir_corwin"] == 1.0


def test_calabrese_kpz():
    assert bench_calabrese_kpz()["synthetic_calabrese_kpz"] == 1.0
