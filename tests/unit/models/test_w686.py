from quant_fund.models.braces_e4 import bench_braces_e4
from quant_fund.models.centralizer_alg import (
    bench_centralizer_alg,
)
from quant_fund.models.delooping2 import bench_delooping2
from quant_fund.models.e4_algebra import bench_e4_algebra
from quant_fund.models.factorization_hom2 import (
    bench_factorization_hom2,
)
from quant_fund.models.koszul_duality2 import (
    bench_koszul_duality2,
)


def test_e4_algebra():
    assert bench_e4_algebra()["synthetic_e4_algebra"] == 1.0


def test_centralizer_alg():
    assert bench_centralizer_alg()["synthetic_centralizer_alg"] == 1.0


def test_delooping2():
    assert bench_delooping2()["synthetic_delooping2"] == 1.0


def test_factorization_hom2():
    assert bench_factorization_hom2()["synthetic_factorization_hom2"] == 1.0


def test_koszul_duality2():
    assert bench_koszul_duality2()["synthetic_koszul_duality2"] == 1.0


def test_braces_e4():
    assert bench_braces_e4()["synthetic_braces_e4"] == 1.0
