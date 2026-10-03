from quant_fund.models.finite_difference import bench_finite_difference
from quant_fund.models.hadamard_matrix import bench_hadamard_matrix
from quant_fund.models.inc_structure import bench_inc_structure
from quant_fund.models.latin_trade import bench_latin_trade
from quant_fund.models.orthogonal_array import bench_orthogonal_array
from quant_fund.models.steiner_system import bench_steiner_system


def test_latin_trade():
    assert bench_latin_trade()["synthetic_latin_trade"] == 1.0


def test_steiner_system():
    assert bench_steiner_system()["synthetic_steiner_system"] == 1.0


def test_inc_structure():
    assert bench_inc_structure()["synthetic_inc_structure"] == 1.0


def test_orthogonal_array():
    assert bench_orthogonal_array()["synthetic_orthogonal_array"] == 1.0


def test_hadamard_matrix():
    assert bench_hadamard_matrix()["synthetic_hadamard_matrix"] == 1.0


def test_finite_difference():
    assert bench_finite_difference()["synthetic_finite_difference"] == 1.0
