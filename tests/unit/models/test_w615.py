from quant_fund.models.extremal_ray import bench_extremal_ray
from quant_fund.models.mori_bir import bench_mori_bir
from quant_fund.models.motivic_adams import bench_motivic_adams
from quant_fund.models.motivic_classifying import (
    bench_motivic_classifying,
)
from quant_fund.models.motivic_dg import bench_motivic_dg
from quant_fund.models.tate_object import bench_tate_object


def test_motivic_adams():
    assert bench_motivic_adams()["synthetic_motivic_adams"] == 1.0


def test_motivic_classifying():
    assert bench_motivic_classifying()["synthetic_motivic_classifying"] == 1.0


def test_tate_object():
    assert bench_tate_object()["synthetic_tate_object"] == 1.0


def test_motivic_dg():
    assert bench_motivic_dg()["synthetic_motivic_dg"] == 1.0


def test_mori_bir():
    assert bench_mori_bir()["synthetic_mori_bir"] == 1.0


def test_extremal_ray():
    assert bench_extremal_ray()["synthetic_extremal_ray"] == 1.0
