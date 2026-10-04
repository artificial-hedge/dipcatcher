from quant_fund.models.earthquake_map import bench_earthquake_map
from quant_fund.models.extremal_length import bench_extremal_length
from quant_fund.models.mapping_class import bench_mapping_class
from quant_fund.models.pseudo_anosov import bench_pseudo_anosov
from quant_fund.models.quadratic_diff import bench_quadratic_diff
from quant_fund.models.weil_petersson import bench_weil_petersson


def test_weil_petersson():
    assert bench_weil_petersson()["synthetic_weil_petersson"] == 1.0


def test_mapping_class():
    assert bench_mapping_class()["synthetic_mapping_class"] == 1.0


def test_quadratic_diff():
    assert bench_quadratic_diff()["synthetic_quadratic_diff"] == 1.0


def test_earthquake_map():
    assert bench_earthquake_map()["synthetic_earthquake_map"] == 1.0


def test_extremal_length():
    assert bench_extremal_length()["synthetic_extremal_length"] == 1.0


def test_pseudo_anosov():
    assert bench_pseudo_anosov()["synthetic_pseudo_anosov"] == 1.0
