from quant_fund.models.bernstein_form import (
    bench_bernstein_form,
)
from quant_fund.models.cardinal_interp import (
    bench_cardinal_interp,
)
from quant_fund.models.chebyshev_interp import (
    bench_chebyshev_interp,
)
from quant_fund.models.osculating_interp import (
    bench_osculating_interp,
)
from quant_fund.models.rational_interp import (
    bench_rational_interp,
)
from quant_fund.models.shanks_trans import (
    bench_shanks_trans,
)


def test_cardinal_interp():
    assert bench_cardinal_interp()["synthetic_cardinal_interp"] == 1.0


def test_bernstein_form():
    assert bench_bernstein_form()["synthetic_bernstein_form"] == 1.0


def test_shanks_trans():
    assert bench_shanks_trans()["synthetic_shanks_trans"] == 1.0


def test_chebyshev_interp():
    assert bench_chebyshev_interp()["synthetic_chebyshev_interp"] == 1.0


def test_osculating_interp():
    assert bench_osculating_interp()["synthetic_osculating_interp"] == 1.0


def test_rational_interp():
    assert bench_rational_interp()["synthetic_rational_interp"] == 1.0
