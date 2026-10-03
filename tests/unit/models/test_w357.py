from quant_fund.models.connection_form import bench_connection_form
from quant_fund.models.gauss_bonnet import bench_gauss_bonnet
from quant_fund.models.geodesic_eq import bench_geodesic_eq
from quant_fund.models.holonomy import bench_holonomy
from quant_fund.models.parallel_transport import bench_parallel_transport
from quant_fund.models.sectional_curv import bench_sectional_curv


def test_connection_form():
    assert bench_connection_form()["synthetic_connection_form"] == 1.0


def test_parallel_transport():
    assert bench_parallel_transport()["synthetic_parallel_transport"] == 1.0


def test_holonomy():
    assert bench_holonomy()["synthetic_holonomy"] == 1.0


def test_gauss_bonnet():
    assert bench_gauss_bonnet()["synthetic_gauss_bonnet"] == 1.0


def test_geodesic_eq():
    assert bench_geodesic_eq()["synthetic_geodesic_eq"] == 1.0


def test_sectional_curv():
    assert bench_sectional_curv()["synthetic_sectional_curv"] == 1.0
