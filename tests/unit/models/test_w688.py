from quant_fund.models.spectral_cellular import (
    bench_spectral_cellular,
)
from quant_fund.models.spectral_cohomological import (
    bench_spectral_cohomological,
)
from quant_fund.models.spectral_field import bench_spectral_field
from quant_fund.models.spectral_filtration import (
    bench_spectral_filtration,
)
from quant_fund.models.spectral_finite import (
    bench_spectral_finite,
)
from quant_fund.models.spectral_lattice import (
    bench_spectral_lattice,
)


def test_spectral_field():
    assert bench_spectral_field()["synthetic_spectral_field"] == 1.0


def test_spectral_lattice():
    assert bench_spectral_lattice()["synthetic_spectral_lattice"] == 1.0


def test_spectral_filtration():
    assert bench_spectral_filtration()["synthetic_spectral_filtration"] == 1.0


def test_spectral_cellular():
    assert bench_spectral_cellular()["synthetic_spectral_cellular"] == 1.0


def test_spectral_cohomological():
    assert bench_spectral_cohomological()["synthetic_spectral_cohomological"] == 1.0


def test_spectral_finite():
    assert bench_spectral_finite()["synthetic_spectral_finite"] == 1.0
