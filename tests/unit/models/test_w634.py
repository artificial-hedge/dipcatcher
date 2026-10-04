from quant_fund.models.bicat2 import bench_bicat2
from quant_fund.models.cat_3cell import bench_cat_3cell
from quant_fund.models.double_lim import bench_double_lim
from quant_fund.models.icon_cat import bench_icon_cat
from quant_fund.models.two_transform import (
    bench_two_transform,
)
from quant_fund.models.vert_cat import bench_vert_cat


def test_icon_cat():
    assert bench_icon_cat()["synthetic_icon_cat"] == 1.0


def test_bicat2():
    assert bench_bicat2()["synthetic_bicat2"] == 1.0


def test_vert_cat():
    assert bench_vert_cat()["synthetic_vert_cat"] == 1.0


def test_double_lim():
    assert bench_double_lim()["synthetic_double_lim"] == 1.0


def test_two_transform():
    assert bench_two_transform()["synthetic_two_transform"] == 1.0


def test_cat_3cell():
    assert bench_cat_3cell()["synthetic_cat_3cell"] == 1.0
