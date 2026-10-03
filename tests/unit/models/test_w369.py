from quant_fund.models.aitken_delta import bench_aitken_delta
from quant_fund.models.brent_root import bench_brent_root
from quant_fund.models.broyden import bench_broyden
from quant_fund.models.cheb_approx import bench_cheb_approx
from quant_fund.models.collocation_ode import bench_collocation_ode
from quant_fund.models.romberg import bench_romberg


def test_broyden():
    assert bench_broyden()["synthetic_broyden"] == 1.0


def test_cheb_approx():
    assert bench_cheb_approx()["synthetic_cheb_approx"] == 1.0


def test_brent_root():
    assert bench_brent_root()["synthetic_brent_root"] == 1.0


def test_romberg():
    assert bench_romberg()["synthetic_romberg"] == 1.0


def test_aitken_delta():
    assert bench_aitken_delta()["synthetic_aitken_delta"] == 1.0


def test_collocation_ode():
    assert bench_collocation_ode()["synthetic_collocation_ode"] == 1.0
