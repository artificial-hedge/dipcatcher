from quant_fund.models.azure_space import (
    bench_azure_space,
)
from quant_fund.models.elliptic_cohom2 import (
    bench_elliptic_cohom2,
)
from quant_fund.models.spectral_etale import (
    bench_spectral_etale,
)
from quant_fund.models.spectral_group import (
    bench_spectral_group,
)
from quant_fund.models.spectral_scheme3 import (
    bench_spectral_scheme3,
)
from quant_fund.models.spectral_smooth import (
    bench_spectral_smooth,
)


def test_spectral_group():
    assert bench_spectral_group()["synthetic_spectral_group"] == 1.0


def test_azure_space():
    assert bench_azure_space()["synthetic_azure_space"] == 1.0


def test_spectral_scheme3():
    assert bench_spectral_scheme3()["synthetic_spectral_scheme3"] == 1.0


def test_spectral_smooth():
    assert bench_spectral_smooth()["synthetic_spectral_smooth"] == 1.0


def test_spectral_etale():
    assert bench_spectral_etale()["synthetic_spectral_etale"] == 1.0


def test_elliptic_cohom2():
    assert bench_elliptic_cohom2()["synthetic_elliptic_cohom2"] == 1.0
