from quant_fund.models.clenshaw_quad import (
    bench_clenshaw_quad,
)
from quant_fund.models.elliptic_fn import (
    bench_elliptic_fn,
)
from quant_fund.models.fejer_nested import (
    bench_fejer_nested,
)
from quant_fund.models.hartley_transform import (
    bench_hartley_transform,
)
from quant_fund.models.radon_transform import (
    bench_radon_transform,
)
from quant_fund.models.zeta_fn import (
    bench_zeta_fn,
)


def test_zeta_fn():
    assert bench_zeta_fn()["synthetic_zeta_fn"] == 1.0


def test_elliptic_fn():
    assert bench_elliptic_fn()["synthetic_elliptic_fn"] == 1.0


def test_hartley_transform():
    assert bench_hartley_transform()["synthetic_hartley_transform"] == 1.0


def test_radon_transform():
    assert bench_radon_transform()["synthetic_radon_transform"] == 1.0


def test_clenshaw_quad():
    assert bench_clenshaw_quad()["synthetic_clenshaw_quad"] == 1.0


def test_fejer_nested():
    assert bench_fejer_nested()["synthetic_fejer_nested"] == 1.0
