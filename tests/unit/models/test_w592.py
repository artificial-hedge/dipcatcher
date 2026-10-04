from quant_fund.models.braided_cat import bench_braided_cat
from quant_fund.models.fusion_cat import bench_fusion_cat
from quant_fund.models.premodular import bench_premodular
from quant_fund.models.rigid_cat import bench_rigid_cat
from quant_fund.models.spherical_cat import (
    bench_spherical_cat,
)
from quant_fund.models.tensor_cat import bench_tensor_cat


def test_tensor_cat():
    assert bench_tensor_cat()["synthetic_tensor_cat"] == 1.0


def test_braided_cat():
    assert bench_braided_cat()["synthetic_braided_cat"] == 1.0


def test_rigid_cat():
    assert bench_rigid_cat()["synthetic_rigid_cat"] == 1.0


def test_fusion_cat():
    assert bench_fusion_cat()["synthetic_fusion_cat"] == 1.0


def test_spherical_cat():
    assert bench_spherical_cat()["synthetic_spherical_cat"] == 1.0


def test_premodular():
    assert bench_premodular()["synthetic_premodular"] == 1.0
