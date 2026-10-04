from quant_fund.models.git_quotient import bench_git_quotient
from quant_fund.models.hilbert_mumford import (
    bench_hilbert_mumford,
)
from quant_fund.models.kirwan_strat import bench_kirwan_strat
from quant_fund.models.luna_slice import bench_luna_slice
from quant_fund.models.moment_polytope import (
    bench_moment_polytope,
)
from quant_fund.models.symplectic_quot import (
    bench_symplectic_quot,
)


def test_git_quotient():
    assert bench_git_quotient()["synthetic_git_quotient"] == 1.0


def test_hilbert_mumford():
    assert bench_hilbert_mumford()["synthetic_hilbert_mumford"] == 1.0


def test_moment_polytope():
    assert bench_moment_polytope()["synthetic_moment_polytope"] == 1.0


def test_kirwan_strat():
    assert bench_kirwan_strat()["synthetic_kirwan_strat"] == 1.0


def test_symplectic_quot():
    assert bench_symplectic_quot()["synthetic_symplectic_quot"] == 1.0


def test_luna_slice():
    assert bench_luna_slice()["synthetic_luna_slice"] == 1.0
