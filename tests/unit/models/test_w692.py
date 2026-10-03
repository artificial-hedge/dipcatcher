from quant_fund.models.centralizer_alg2 import (
    bench_centralizer_alg2,
)
from quant_fund.models.e5_algebra import bench_e5_algebra
from quant_fund.models.factorization_hom3 import (
    bench_factorization_hom3,
)
from quant_fund.models.framed_discs import (
    bench_framed_discs,
)
from quant_fund.models.little_cubes2 import (
    bench_little_cubes2,
)
from quant_fund.models.swiss_cheese3 import (
    bench_swiss_cheese3,
)


def test_e5_algebra():
    assert bench_e5_algebra()["synthetic_e5_algebra"] == 1.0


def test_little_cubes2():
    assert bench_little_cubes2()["synthetic_little_cubes2"] == 1.0


def test_swiss_cheese3():
    assert bench_swiss_cheese3()["synthetic_swiss_cheese3"] == 1.0


def test_framed_discs():
    assert bench_framed_discs()["synthetic_framed_discs"] == 1.0


def test_factorization_hom3():
    assert bench_factorization_hom3()["synthetic_factorization_hom3"] == 1.0


def test_centralizer_alg2():
    assert bench_centralizer_alg2()["synthetic_centralizer_alg2"] == 1.0
