from quant_fund.models.adic_generic import bench_adic_generic
from quant_fund.models.dagger_space import bench_dagger_space
from quant_fund.models.fargues_curve import bench_fargues_curve
from quant_fund.models.huber_ring import bench_huber_ring
from quant_fund.models.prism_site import bench_prism_site
from quant_fund.models.witt_perfect import bench_witt_perfect


def test_dagger_space():
    assert bench_dagger_space()["synthetic_dagger_space"] == 1.0


def test_huber_ring():
    assert bench_huber_ring()["synthetic_huber_ring"] == 1.0


def test_adic_generic():
    assert bench_adic_generic()["synthetic_adic_generic"] == 1.0


def test_witt_perfect():
    assert bench_witt_perfect()["synthetic_witt_perfect"] == 1.0


def test_fargues_curve():
    assert bench_fargues_curve()["synthetic_fargues_curve"] == 1.0


def test_prism_site():
    assert bench_prism_site()["synthetic_prism_site"] == 1.0
