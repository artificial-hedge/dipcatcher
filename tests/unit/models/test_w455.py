from quant_fund.models.cubical_path import bench_cubical_path
from quant_fund.models.glue_types import bench_glue_types
from quant_fund.models.hcomp_fill import bench_hcomp_fill
from quant_fund.models.interval_obj import bench_interval_obj
from quant_fund.models.kan_op import bench_kan_op
from quant_fund.models.transport_coe import bench_transport_coe


def test_cubical_path():
    assert bench_cubical_path()["synthetic_cubical_path"] == 1.0


def test_hcomp_fill():
    assert bench_hcomp_fill()["synthetic_hcomp_fill"] == 1.0


def test_glue_types():
    assert bench_glue_types()["synthetic_glue_types"] == 1.0


def test_interval_obj():
    assert bench_interval_obj()["synthetic_interval_obj"] == 1.0


def test_kan_op():
    assert bench_kan_op()["synthetic_kan_op"] == 1.0


def test_transport_coe():
    assert bench_transport_coe()["synthetic_transport_coe"] == 1.0
