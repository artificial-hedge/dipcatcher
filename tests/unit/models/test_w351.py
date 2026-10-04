from quant_fund.models.banach_fixed import bench_banach_fixed
from quant_fund.models.compact_operator import bench_compact_operator
from quant_fund.models.fourier_finite import bench_fourier_finite
from quant_fund.models.gram_schmidt import bench_gram_schmidt
from quant_fund.models.lp_duality import bench_lp_duality
from quant_fund.models.spectral_theorem import bench_spectral_theorem


def test_banach_fixed():
    assert bench_banach_fixed()["synthetic_banach_fixed"] == 1.0


def test_spectral_theorem():
    assert bench_spectral_theorem()["synthetic_spectral_theorem"] == 1.0


def test_lp_duality():
    assert bench_lp_duality()["synthetic_lp_duality"] == 1.0


def test_fourier_finite():
    assert bench_fourier_finite()["synthetic_fourier_finite"] == 1.0


def test_compact_operator():
    assert bench_compact_operator()["synthetic_compact_operator"] == 1.0


def test_gram_schmidt():
    assert bench_gram_schmidt()["synthetic_gram_schmidt"] == 1.0
