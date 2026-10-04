from quant_fund.models.area_mart import bench_area_mart
from quant_fund.models.controlled_path import (
    bench_controlled_path,
)
from quant_fund.models.hairspring_map import (
    bench_hairspring_map,
)
from quant_fund.models.lyons_lift import (
    bench_lyons_lift,
)
from quant_fund.models.rough_path import bench_rough_path
from quant_fund.models.signature_transform2 import (
    bench_signature_transform2,
)


def test_rough_path():
    assert bench_rough_path()["synthetic_rough_path"] == 1.0


def test_signature_transform2():
    assert bench_signature_transform2()["synthetic_signature_transform2"] == 1.0


def test_controlled_path():
    assert bench_controlled_path()["synthetic_controlled_path"] == 1.0


def test_lyons_lift():
    assert bench_lyons_lift()["synthetic_lyons_lift"] == 1.0


def test_hairspring_map():
    assert bench_hairspring_map()["synthetic_hairspring_map"] == 1.0


def test_area_mart():
    assert bench_area_mart()["synthetic_area_mart"] == 1.0
