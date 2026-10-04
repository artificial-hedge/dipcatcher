from quant_fund.models.airy_fn import (
    bench_airy_fn,
)
from quant_fund.models.bessel_fn import (
    bench_bessel_fn,
)
from quant_fund.models.beta_fn import (
    bench_beta_fn,
)
from quant_fund.models.error_fn import (
    bench_error_fn,
)
from quant_fund.models.gamma_fn import (
    bench_gamma_fn,
)
from quant_fund.models.hypergeometric_fn import (
    bench_hypergeometric_fn,
)


def test_gamma_fn():
    assert bench_gamma_fn()["synthetic_gamma_fn"] == 1.0


def test_beta_fn():
    assert bench_beta_fn()["synthetic_beta_fn"] == 1.0


def test_bessel_fn():
    assert bench_bessel_fn()["synthetic_bessel_fn"] == 1.0


def test_airy_fn():
    assert bench_airy_fn()["synthetic_airy_fn"] == 1.0


def test_error_fn():
    assert bench_error_fn()["synthetic_error_fn"] == 1.0


def test_hypergeometric_fn():
    assert bench_hypergeometric_fn()["synthetic_hypergeometric_fn"] == 1.0
