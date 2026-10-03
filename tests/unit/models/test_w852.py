from quant_fund.models.bem_kernel import (
    bench_bem_kernel,
)
from quant_fund.models.fast_multipole import (
    bench_fast_multipole,
)
from quant_fund.models.fredholm_solve import (
    bench_fredholm_solve,
)
from quant_fund.models.galerkin_bem import (
    bench_galerkin_bem,
)
from quant_fund.models.nystrom_method import (
    bench_nystrom_method,
)
from quant_fund.models.singular_integrals import (
    bench_singular_integrals,
)


def test_bem_kernel():
    assert bench_bem_kernel()["synthetic_bem_kernel"] == 1.0


def test_fredholm_solve():
    assert bench_fredholm_solve()["synthetic_fredholm_solve"] == 1.0


def test_nystrom_method():
    assert bench_nystrom_method()["synthetic_nystrom_method"] == 1.0


def test_singular_integrals():
    assert bench_singular_integrals()["synthetic_singular_integrals"] == 1.0


def test_fast_multipole():
    assert bench_fast_multipole()["synthetic_fast_multipole"] == 1.0


def test_galerkin_bem():
    assert bench_galerkin_bem()["synthetic_galerkin_bem"] == 1.0
