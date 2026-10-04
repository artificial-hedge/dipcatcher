from quant_fund.models.spectral_dedekind import (
    bench_spectral_dedekind,
)
from quant_fund.models.spectral_dvr import bench_spectral_dvr
from quant_fund.models.spectral_excellent import (
    bench_spectral_excellent,
)
from quant_fund.models.spectral_jacobson import (
    bench_spectral_jacobson,
)
from quant_fund.models.spectral_noether import (
    bench_spectral_noether,
)
from quant_fund.models.spectral_regular import (
    bench_spectral_regular,
)


def test_spectral_dvr():
    assert bench_spectral_dvr()["synthetic_spectral_dvr"] == 1.0


def test_spectral_noether():
    assert bench_spectral_noether()["synthetic_spectral_noether"] == 1.0


def test_spectral_regular():
    assert bench_spectral_regular()["synthetic_spectral_regular"] == 1.0


def test_spectral_dedekind():
    assert bench_spectral_dedekind()["synthetic_spectral_dedekind"] == 1.0


def test_spectral_jacobson():
    assert bench_spectral_jacobson()["synthetic_spectral_jacobson"] == 1.0


def test_spectral_excellent():
    assert bench_spectral_excellent()["synthetic_spectral_excellent"] == 1.0
