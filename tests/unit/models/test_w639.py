from quant_fund.models.finite_chromatic import (
    bench_finite_chromatic,
)
from quant_fund.models.finite_htpy import (
    bench_finite_htpy,
)
from quant_fund.models.homotopy_fiber2 import (
    bench_homotopy_fiber2,
)
from quant_fund.models.periodic_htpy import (
    bench_periodic_htpy,
)
from quant_fund.models.rational_spec import (
    bench_rational_spec,
)
from quant_fund.models.stable_htpy2 import (
    bench_stable_htpy2,
)


def test_homotopy_fiber2():
    assert bench_homotopy_fiber2()["synthetic_homotopy_fiber2"] == 1.0


def test_stable_htpy2():
    assert bench_stable_htpy2()["synthetic_stable_htpy2"] == 1.0


def test_finite_htpy():
    assert bench_finite_htpy()["synthetic_finite_htpy"] == 1.0


def test_rational_spec():
    assert bench_rational_spec()["synthetic_rational_spec"] == 1.0


def test_finite_chromatic():
    assert bench_finite_chromatic()["synthetic_finite_chromatic"] == 1.0


def test_periodic_htpy():
    assert bench_periodic_htpy()["synthetic_periodic_htpy"] == 1.0
