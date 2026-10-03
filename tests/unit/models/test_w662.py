from quant_fund.models.bar_resolution2 import (
    bench_bar_resolution2,
)
from quant_fund.models.braces_higher import bench_braces_higher
from quant_fund.models.deligne_conj2 import bench_deligne_conj2
from quant_fund.models.factor_homology2 import (
    bench_factor_homology2,
)
from quant_fund.models.hochschild_hom2 import (
    bench_hochschild_hom2,
)
from quant_fund.models.little_cubes import bench_little_cubes


def test_bar_resolution2():
    assert bench_bar_resolution2()["synthetic_bar_resolution2"] == 1.0


def test_hochschild_hom2():
    assert bench_hochschild_hom2()["synthetic_hochschild_hom2"] == 1.0


def test_factor_homology2():
    assert bench_factor_homology2()["synthetic_factor_homology2"] == 1.0


def test_deligne_conj2():
    assert bench_deligne_conj2()["synthetic_deligne_conj2"] == 1.0


def test_braces_higher():
    assert bench_braces_higher()["synthetic_braces_higher"] == 1.0


def test_little_cubes():
    assert bench_little_cubes()["synthetic_little_cubes"] == 1.0
