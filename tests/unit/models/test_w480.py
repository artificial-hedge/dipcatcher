from quant_fund.models.classify_obj import bench_classify_obj
from quant_fund.models.etale_geom import bench_etale_geom
from quant_fund.models.exponentiable import bench_exponentiable
from quant_fund.models.gros_topos import bench_gros_topos
from quant_fund.models.local_homeo import bench_local_homeo
from quant_fund.models.pi_infty import bench_pi_infty


def test_etale_geom():
    assert bench_etale_geom()["synthetic_etale_geom"] == 1.0


def test_gros_topos():
    assert bench_gros_topos()["synthetic_gros_topos"] == 1.0


def test_local_homeo():
    assert bench_local_homeo()["synthetic_local_homeo"] == 1.0


def test_classify_obj():
    assert bench_classify_obj()["synthetic_classify_obj"] == 1.0


def test_pi_infty():
    assert bench_pi_infty()["synthetic_pi_infty"] == 1.0


def test_exponentiable():
    assert bench_exponentiable()["synthetic_exponentiable"] == 1.0
