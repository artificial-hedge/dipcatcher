from quant_fund.models.spectral_coord import bench_spectral_coord
from quant_fund.models.spectral_ext_field import (
    bench_spectral_ext_field,
)
from quant_fund.models.spectral_level import bench_spectral_level
from quant_fund.models.spectral_polynomial2 import (
    bench_spectral_polynomial2,
)
from quant_fund.models.spectral_prime import bench_spectral_prime
from quant_fund.models.spectral_residue import (
    bench_spectral_residue,
)


def test_spectral_prime():
    assert bench_spectral_prime()["synthetic_spectral_prime"] == 1.0


def test_spectral_residue():
    assert bench_spectral_residue()["synthetic_spectral_residue"] == 1.0


def test_spectral_level():
    assert bench_spectral_level()["synthetic_spectral_level"] == 1.0


def test_spectral_polynomial2():
    assert bench_spectral_polynomial2()["synthetic_spectral_polynomial2"] == 1.0


def test_spectral_coord():
    assert bench_spectral_coord()["synthetic_spectral_coord"] == 1.0


def test_spectral_ext_field():
    assert bench_spectral_ext_field()["synthetic_spectral_ext_field"] == 1.0
