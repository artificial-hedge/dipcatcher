from quant_fund.models.boolean_algebra import bench_boolean_algebra
from quant_fund.models.congruence_lattice import bench_congruence_lattice
from quant_fund.models.galois_connection import bench_galois_connection
from quant_fund.models.lattice_check import bench_lattice_check
from quant_fund.models.tarski_fixed import bench_tarski_fixed
from quant_fund.models.term_algebra import bench_term_algebra


def test_lattice_check():
    assert bench_lattice_check()["synthetic_lattice_check"] == 1.0


def test_galois_connection():
    assert bench_galois_connection()["synthetic_galois_connection"] == 1.0


def test_tarski_fixed():
    assert bench_tarski_fixed()["synthetic_tarski_fixed"] == 1.0


def test_boolean_algebra():
    assert bench_boolean_algebra()["synthetic_boolean_algebra"] == 1.0


def test_congruence_lattice():
    assert bench_congruence_lattice()["synthetic_congruence_lattice"] == 1.0


def test_term_algebra():
    assert bench_term_algebra()["synthetic_term_algebra"] == 1.0
