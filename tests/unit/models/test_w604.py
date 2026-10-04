from quant_fund.models.bokstedt_periodicity import (
    bench_bokstedt_periodicity,
)
from quant_fund.models.elliptic_k import bench_elliptic_k
from quant_fund.models.equivariant_cohomology2 import (
    bench_equivariant_cohomology2,
)
from quant_fund.models.may_ss import bench_may_ss
from quant_fund.models.topo_k_theory import (
    bench_topo_k_theory,
)
from quant_fund.models.unstable_cohomology import (
    bench_unstable_cohomology,
)


def test_unstable_cohomology():
    assert bench_unstable_cohomology()["synthetic_unstable_cohomology"] == 1.0


def test_may_ss():
    assert bench_may_ss()["synthetic_may_ss"] == 1.0


def test_bokstedt_periodicity():
    assert bench_bokstedt_periodicity()["synthetic_bokstedt_periodicity"] == 1.0


def test_topo_k_theory():
    assert bench_topo_k_theory()["synthetic_topo_k_theory"] == 1.0


def test_elliptic_k():
    assert bench_elliptic_k()["synthetic_elliptic_k"] == 1.0


def test_equivariant_cohomology2():
    assert bench_equivariant_cohomology2()["synthetic_equivariant_cohomology2"] == 1.0
