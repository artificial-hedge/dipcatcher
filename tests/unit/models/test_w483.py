from quant_fund.models.bar_spec import bench_bar_spec
from quant_fund.models.dyer_lashof import bench_dyer_lashof
from quant_fund.models.free_loop import bench_free_loop
from quant_fund.models.loop_functor import bench_loop_functor
from quant_fund.models.steenrod_ops import bench_steenrod_ops
from quant_fund.models.sullivan_min import bench_sullivan_min


def test_steenrod_ops():
    assert bench_steenrod_ops()["synthetic_steenrod_ops"] == 1.0


def test_dyer_lashof():
    assert bench_dyer_lashof()["synthetic_dyer_lashof"] == 1.0


def test_bar_spec():
    assert bench_bar_spec()["synthetic_bar_spec"] == 1.0


def test_free_loop():
    assert bench_free_loop()["synthetic_free_loop"] == 1.0


def test_sullivan_min():
    assert bench_sullivan_min()["synthetic_sullivan_min"] == 1.0


def test_loop_functor():
    assert bench_loop_functor()["synthetic_loop_functor"] == 1.0
