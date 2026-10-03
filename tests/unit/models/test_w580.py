from quant_fund.models.cycle_index import bench_cycle_index
from quant_fund.models.exponential_gf import (
    bench_exponential_gf,
)
from quant_fund.models.lagrange_inversion import (
    bench_lagrange_inversion,
)
from quant_fund.models.matrix_tree import bench_matrix_tree
from quant_fund.models.species_theory import (
    bench_species_theory,
)
from quant_fund.models.transfer_matrix import (
    bench_transfer_matrix,
)


def test_species_theory():
    assert bench_species_theory()["synthetic_species_theory"] == 1.0


def test_cycle_index():
    assert bench_cycle_index()["synthetic_cycle_index"] == 1.0


def test_lagrange_inversion():
    assert bench_lagrange_inversion()["synthetic_lagrange_inversion"] == 1.0


def test_transfer_matrix():
    assert bench_transfer_matrix()["synthetic_transfer_matrix"] == 1.0


def test_matrix_tree():
    assert bench_matrix_tree()["synthetic_matrix_tree"] == 1.0


def test_exponential_gf():
    assert bench_exponential_gf()["synthetic_exponential_gf"] == 1.0
