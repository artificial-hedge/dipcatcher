from quant_fund.models.artins_theorem import bench_artins_theorem
from quant_fund.models.clifford_toy import bench_clifford_toy
from quant_fund.models.frobenius_group import bench_frobenius_group
from quant_fund.models.induced_char import bench_induced_char
from quant_fund.models.schur_index import bench_schur_index
from quant_fund.models.tensor_char import bench_tensor_char


def test_induced_char():
    assert bench_induced_char()["synthetic_induced_char"] == 1.0


def test_artins_theorem():
    assert bench_artins_theorem()["synthetic_artins_theorem"] == 1.0


def test_tensor_char():
    assert bench_tensor_char()["synthetic_tensor_char"] == 1.0


def test_clifford_toy():
    assert bench_clifford_toy()["synthetic_clifford_toy"] == 1.0


def test_schur_index():
    assert bench_schur_index()["synthetic_schur_index"] == 1.0


def test_frobenius_group():
    assert bench_frobenius_group()["synthetic_frobenius_group"] == 1.0
