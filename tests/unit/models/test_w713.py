from quant_fund.models.derived_affine import (
    bench_derived_affine,
)
from quant_fund.models.derived_projective import (
    bench_derived_projective,
)
from quant_fund.models.spectral_artin import (
    bench_spectral_artin,
)
from quant_fund.models.spectral_dirac import (
    bench_spectral_dirac,
)
from quant_fund.models.spectral_gal import bench_spectral_gal
from quant_fund.models.spectral_semi import bench_spectral_semi


def test_spectral_semi():
    assert bench_spectral_semi()["synthetic_spectral_semi"] == 1.0


def test_spectral_artin():
    assert bench_spectral_artin()["synthetic_spectral_artin"] == 1.0


def test_spectral_gal():
    assert bench_spectral_gal()["synthetic_spectral_gal"] == 1.0


def test_spectral_dirac():
    assert bench_spectral_dirac()["synthetic_spectral_dirac"] == 1.0


def test_derived_affine():
    assert bench_derived_affine()["synthetic_derived_affine"] == 1.0


def test_derived_projective():
    assert bench_derived_projective()["synthetic_derived_projective"] == 1.0
