from quant_fund.models.ample_test import bench_ample_test
from quant_fund.models.chow_ring import bench_chow_ring
from quant_fund.models.grothendieck_grp import bench_grothendieck_grp
from quant_fund.models.gysin import bench_gysin
from quant_fund.models.proj_morph import bench_proj_morph
from quant_fund.models.toric_variety import bench_toric_variety


def test_grothendieck_grp():
    assert bench_grothendieck_grp()["synthetic_grothendieck_grp"] == 1.0


def test_chow_ring():
    assert bench_chow_ring()["synthetic_chow_ring"] == 1.0


def test_gysin():
    assert bench_gysin()["synthetic_gysin"] == 1.0


def test_toric_variety():
    assert bench_toric_variety()["synthetic_toric_variety"] == 1.0


def test_proj_morph():
    assert bench_proj_morph()["synthetic_proj_morph"] == 1.0


def test_ample_test():
    assert bench_ample_test()["synthetic_ample_test"] == 1.0
