from quant_fund.models.exceptional_coll import (
    bench_exceptional_coll,
)
from quant_fund.models.fourier_mukai import (
    bench_fourier_mukai,
)
from quant_fund.models.semi_orthogonal import (
    bench_semi_orthogonal,
)
from quant_fund.models.serre_functor import (
    bench_serre_functor,
)
from quant_fund.models.sod_decomp import bench_sod_decomp
from quant_fund.models.spherical_functor import (
    bench_spherical_functor,
)


def test_exceptional_coll():
    assert bench_exceptional_coll()["synthetic_exceptional_coll"] == 1.0


def test_spherical_functor():
    assert bench_spherical_functor()["synthetic_spherical_functor"] == 1.0


def test_serre_functor():
    assert bench_serre_functor()["synthetic_serre_functor"] == 1.0


def test_sod_decomp():
    assert bench_sod_decomp()["synthetic_sod_decomp"] == 1.0


def test_fourier_mukai():
    assert bench_fourier_mukai()["synthetic_fourier_mukai"] == 1.0


def test_semi_orthogonal():
    assert bench_semi_orthogonal()["synthetic_semi_orthogonal"] == 1.0
