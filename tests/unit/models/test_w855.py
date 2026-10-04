from quant_fund.models.adapt_wavelet import (
    bench_adapt_wavelet,
)
from quant_fund.models.coiflet_basis import (
    bench_coiflet_basis,
)
from quant_fund.models.daubechies_basis import (
    bench_daubechies_basis,
)
from quant_fund.models.spline_wavelet import (
    bench_spline_wavelet,
)
from quant_fund.models.wavelet_collocation import (
    bench_wavelet_collocation,
)
from quant_fund.models.wavelet_galerkin import (
    bench_wavelet_galerkin,
)


def test_wavelet_galerkin():
    assert bench_wavelet_galerkin()["synthetic_wavelet_galerkin"] == 1.0


def test_daubechies_basis():
    assert bench_daubechies_basis()["synthetic_daubechies_basis"] == 1.0


def test_coiflet_basis():
    assert bench_coiflet_basis()["synthetic_coiflet_basis"] == 1.0


def test_spline_wavelet():
    assert bench_spline_wavelet()["synthetic_spline_wavelet"] == 1.0


def test_wavelet_collocation():
    assert bench_wavelet_collocation()["synthetic_wavelet_collocation"] == 1.0


def test_adapt_wavelet():
    assert bench_adapt_wavelet()["synthetic_adapt_wavelet"] == 1.0
