from quant_fund.models.chebyshev_collocation import (
    bench_chebyshev_collocation,
)
from quant_fund.models.chebyshev_grid import (
    bench_chebyshev_grid,
)
from quant_fund.models.dealiasing import (
    bench_dealiasing,
)
from quant_fund.models.fourier_galerkin import (
    bench_fourier_galerkin,
)
from quant_fund.models.legendre_tau import (
    bench_legendre_tau,
)
from quant_fund.models.spectral_deriv import (
    bench_spectral_deriv,
)


def test_chebyshev_grid():
    assert bench_chebyshev_grid()["synthetic_chebyshev_grid"] == 1.0


def test_fourier_galerkin():
    assert bench_fourier_galerkin()["synthetic_fourier_galerkin"] == 1.0


def test_legendre_tau():
    assert bench_legendre_tau()["synthetic_legendre_tau"] == 1.0


def test_chebyshev_collocation():
    assert bench_chebyshev_collocation()["synthetic_chebyshev_collocation"] == 1.0


def test_spectral_deriv():
    assert bench_spectral_deriv()["synthetic_spectral_deriv"] == 1.0


def test_dealiasing():
    assert bench_dealiasing()["synthetic_dealiasing"] == 1.0
