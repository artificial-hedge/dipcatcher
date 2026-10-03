from quant_fund.models.crystal_base import bench_crystal_base
from quant_fund.models.jimbo_drin import bench_jimbo_drin
from quant_fund.models.lusztig_can import bench_lusztig_can
from quant_fund.models.quantum_group import bench_quantum_group
from quant_fund.models.quantum_rmatrix import bench_quantum_rmatrix
from quant_fund.models.quantum_schur import bench_quantum_schur


def test_quantum_group():
    assert bench_quantum_group()["synthetic_quantum_group"] == 1.0


def test_crystal_base():
    assert bench_crystal_base()["synthetic_crystal_base"] == 1.0


def test_quantum_rmatrix():
    assert bench_quantum_rmatrix()["synthetic_quantum_rmatrix"] == 1.0


def test_jimbo_drin():
    assert bench_jimbo_drin()["synthetic_jimbo_drin"] == 1.0


def test_lusztig_can():
    assert bench_lusztig_can()["synthetic_lusztig_can"] == 1.0


def test_quantum_schur():
    assert bench_quantum_schur()["synthetic_quantum_schur"] == 1.0
