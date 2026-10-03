from quant_fund.models.spectral_abelian import bench_spectral_abelian
from quant_fund.models.spectral_crystal import bench_spectral_crystal
from quant_fund.models.spectral_etale2 import bench_spectral_etale2
from quant_fund.models.spectral_perfect import bench_spectral_perfect
from quant_fund.models.spectral_proper import bench_spectral_proper
from quant_fund.models.spectral_smooth2 import bench_spectral_smooth2


def test_spectral_perfect():
    assert bench_spectral_perfect()["synthetic_spectral_perfect"] == 1.0


def test_spectral_smooth2():
    assert bench_spectral_smooth2()["synthetic_spectral_smooth2"] == 1.0


def test_spectral_etale2():
    assert bench_spectral_etale2()["synthetic_spectral_etale2"] == 1.0


def test_spectral_abelian():
    assert bench_spectral_abelian()["synthetic_spectral_abelian"] == 1.0


def test_spectral_crystal():
    assert bench_spectral_crystal()["synthetic_spectral_crystal"] == 1.0


def test_spectral_proper():
    assert bench_spectral_proper()["synthetic_spectral_proper"] == 1.0
