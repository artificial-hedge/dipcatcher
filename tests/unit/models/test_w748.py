from quant_fund.models.aggarwal_sixv import bench_aggarwal_sixv
from quant_fund.models.baxter_vertex import bench_baxter_vertex
from quant_fund.models.borodin_sixv import bench_borodin_sixv
from quant_fund.models.corwin_petrov import bench_corwin_petrov
from quant_fund.models.gowers_knot import bench_gowers_knot
from quant_fund.models.reshetikhin_vertex import (
    bench_reshetikhin_vertex,
)


def test_borodin_sixv():
    assert bench_borodin_sixv()["synthetic_borodin_sixv"] == 1.0


def test_gowers_knot():
    assert bench_gowers_knot()["synthetic_gowers_knot"] == 1.0


def test_baxter_vertex():
    assert bench_baxter_vertex()["synthetic_baxter_vertex"] == 1.0


def test_reshetikhin_vertex():
    assert bench_reshetikhin_vertex()["synthetic_reshetikhin_vertex"] == 1.0


def test_corwin_petrov():
    assert bench_corwin_petrov()["synthetic_corwin_petrov"] == 1.0


def test_aggarwal_sixv():
    assert bench_aggarwal_sixv()["synthetic_aggarwal_sixv"] == 1.0
