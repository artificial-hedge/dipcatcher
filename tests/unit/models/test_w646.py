from quant_fund.models.bdr_plus import bench_bdr_plus
from quant_fund.models.diamond_sheaf import (
    bench_diamond_sheaf,
)
from quant_fund.models.fargues_cat import bench_fargues_cat
from quant_fund.models.spatial_diamond import (
    bench_spatial_diamond,
)
from quant_fund.models.untilt2 import bench_untilt2
from quant_fund.models.v_stack import bench_v_stack


def test_fargues_cat():
    assert bench_fargues_cat()["synthetic_fargues_cat"] == 1.0


def test_v_stack():
    assert bench_v_stack()["synthetic_v_stack"] == 1.0


def test_untilt2():
    assert bench_untilt2()["synthetic_untilt2"] == 1.0


def test_spatial_diamond():
    assert bench_spatial_diamond()["synthetic_spatial_diamond"] == 1.0


def test_diamond_sheaf():
    assert bench_diamond_sheaf()["synthetic_diamond_sheaf"] == 1.0


def test_bdr_plus():
    assert bench_bdr_plus()["synthetic_bdr_plus"] == 1.0
