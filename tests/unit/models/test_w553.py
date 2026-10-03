from quant_fund.models.frobenius_mfd import bench_frobenius_mfd
from quant_fund.models.givental_j import bench_givental_j
from quant_fund.models.mirror_symmetry import bench_mirror_symmetry
from quant_fund.models.quantum_cohomology import bench_quantum_cohomology
from quant_fund.models.quintic_invariants import bench_quintic_invariants
from quant_fund.models.toric_mirror import bench_toric_mirror


def test_mirror_symmetry():
    assert bench_mirror_symmetry()["synthetic_mirror_symmetry"] == 1.0


def test_givental_j():
    assert bench_givental_j()["synthetic_givental_j"] == 1.0


def test_quantum_cohomology():
    assert bench_quantum_cohomology()["synthetic_quantum_cohomology"] == 1.0


def test_quintic_invariants():
    assert bench_quintic_invariants()["synthetic_quintic_invariants"] == 1.0


def test_toric_mirror():
    assert bench_toric_mirror()["synthetic_toric_mirror"] == 1.0


def test_frobenius_mfd():
    assert bench_frobenius_mfd()["synthetic_frobenius_mfd"] == 1.0
