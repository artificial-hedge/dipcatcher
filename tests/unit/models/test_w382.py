from quant_fund.models.back_forth import bench_back_forth
from quant_fund.models.indiscernibles import bench_indiscernibles
from quant_fund.models.omitting_types import bench_omitting_types
from quant_fund.models.saturation_test import bench_saturation_test
from quant_fund.models.stability_spec import bench_stability_spec
from quant_fund.models.stone_duality import bench_stone_duality


def test_stone_duality():
    assert bench_stone_duality()["synthetic_stone_duality"] == 1.0


def test_saturation_test():
    assert bench_saturation_test()["synthetic_saturation_test"] == 1.0


def test_omitting_types():
    assert bench_omitting_types()["synthetic_omitting_types"] == 1.0


def test_indiscernibles():
    assert bench_indiscernibles()["synthetic_indiscernibles"] == 1.0


def test_stability_spec():
    assert bench_stability_spec()["synthetic_stability_spec"] == 1.0


def test_back_forth():
    assert bench_back_forth()["synthetic_back_forth"] == 1.0
