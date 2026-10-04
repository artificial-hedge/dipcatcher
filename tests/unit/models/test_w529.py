from quant_fund.models.arnold_diff import bench_arnold_diff
from quant_fund.models.aubry_mather import bench_aubry_mather
from quant_fund.models.cantorus import bench_cantorus
from quant_fund.models.greene_crit import bench_greene_crit
from quant_fund.models.kam_theorem import bench_kam_theorem
from quant_fund.models.twist_map import bench_twist_map


def test_kam_theorem():
    assert bench_kam_theorem()["synthetic_kam_theorem"] == 1.0


def test_aubry_mather():
    assert bench_aubry_mather()["synthetic_aubry_mather"] == 1.0


def test_twist_map():
    assert bench_twist_map()["synthetic_twist_map"] == 1.0


def test_cantorus():
    assert bench_cantorus()["synthetic_cantorus"] == 1.0


def test_greene_crit():
    assert bench_greene_crit()["synthetic_greene_crit"] == 1.0


def test_arnold_diff():
    assert bench_arnold_diff()["synthetic_arnold_diff"] == 1.0
