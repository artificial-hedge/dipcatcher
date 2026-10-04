from quant_fund.models.braided_functor import bench_braided_functor
from quant_fund.models.center_cat import bench_center_cat
from quant_fund.models.ds_category import bench_ds_category
from quant_fund.models.fusion_ring import bench_fusion_ring
from quant_fund.models.multifusion import bench_multifusion
from quant_fund.models.premodular2 import bench_premodular2


def test_multifusion():
    assert bench_multifusion()["synthetic_multifusion"] == 1.0


def test_premodular2():
    assert bench_premodular2()["synthetic_premodular2"] == 1.0


def test_braided_functor():
    assert bench_braided_functor()["synthetic_braided_functor"] == 1.0


def test_center_cat():
    assert bench_center_cat()["synthetic_center_cat"] == 1.0


def test_fusion_ring():
    assert bench_fusion_ring()["synthetic_fusion_ring"] == 1.0


def test_ds_category():
    assert bench_ds_category()["synthetic_ds_category"] == 1.0
