from quant_fund.models.fejer_kernel import bench_fejer_kernel
from quant_fund.models.fourier_multiplier import bench_fourier_multiplier
from quant_fund.models.plancherel import bench_plancherel
from quant_fund.models.poisson_summation import bench_poisson_summation
from quant_fund.models.sobolev_embed import bench_sobolev_embed
from quant_fund.models.uncertainty import bench_uncertainty


def test_plancherel():
    assert bench_plancherel()["synthetic_plancherel"] == 1.0


def test_poisson_summation():
    assert bench_poisson_summation()["synthetic_poisson_summation"] == 1.0


def test_fejer_kernel():
    assert bench_fejer_kernel()["synthetic_fejer_kernel"] == 1.0


def test_uncertainty():
    assert bench_uncertainty()["synthetic_uncertainty"] == 1.0


def test_fourier_multiplier():
    assert bench_fourier_multiplier()["synthetic_fourier_multiplier"] == 1.0


def test_sobolev_embed():
    assert bench_sobolev_embed()["synthetic_sobolev_embed"] == 1.0
