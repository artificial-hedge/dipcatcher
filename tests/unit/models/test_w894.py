from quant_fund.models.barycentric_wts import (
    bench_barycentric_wts,
)
from quant_fund.models.divid_diff_table import (
    bench_divid_diff_table,
)
from quant_fund.models.floater_hormann import (
    bench_floater_hormann,
)
from quant_fund.models.hermite_interp import (
    bench_hermite_interp,
)
from quant_fund.models.lagrange_interp import (
    bench_lagrange_interp,
)
from quant_fund.models.neville_interp import (
    bench_neville_interp,
)


def test_lagrange_interp():
    assert bench_lagrange_interp()["synthetic_lagrange_interp"] == 1.0


def test_neville_interp():
    assert bench_neville_interp()["synthetic_neville_interp"] == 1.0


def test_hermite_interp():
    assert bench_hermite_interp()["synthetic_hermite_interp"] == 1.0


def test_divid_diff_table():
    assert bench_divid_diff_table()["synthetic_divid_diff_table"] == 1.0


def test_barycentric_wts():
    assert bench_barycentric_wts()["synthetic_barycentric_wts"] == 1.0


def test_floater_hormann():
    assert bench_floater_hormann()["synthetic_floater_hormann"] == 1.0
