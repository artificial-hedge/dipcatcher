from quant_fund.models.connective_e_ring import bench_connective_e_ring
from quant_fund.models.elliptic_cohor import bench_elliptic_cohor
from quant_fund.models.spectral_alg import bench_spectral_alg
from quant_fund.models.spectral_scheme2 import bench_spectral_scheme2
from quant_fund.models.spectral_stack import bench_spectral_stack
from quant_fund.models.taf_lurie import bench_taf_lurie


def test_spectral_scheme2():
    assert bench_spectral_scheme2()["synthetic_spectral_scheme2"] == 1.0


def test_connective_e_ring():
    assert bench_connective_e_ring()["synthetic_connective_e_ring"] == 1.0


def test_spectral_alg():
    assert bench_spectral_alg()["synthetic_spectral_alg"] == 1.0


def test_spectral_stack():
    assert bench_spectral_stack()["synthetic_spectral_stack"] == 1.0


def test_elliptic_cohor():
    assert bench_elliptic_cohor()["synthetic_elliptic_cohor"] == 1.0


def test_taf_lurie():
    assert bench_taf_lurie()["synthetic_taf_lurie"] == 1.0
