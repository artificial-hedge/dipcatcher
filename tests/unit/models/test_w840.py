from quant_fund.models.loewner_interp import (
    bench_loewner_interp,
)
from quant_fund.models.nevanlinna_pick import (
    bench_nevanlinna_pick,
)
from quant_fund.models.pade_approx import (
    bench_pade_approx,
)
from quant_fund.models.rational_chebyshev import (
    bench_rational_chebyshev,
)
from quant_fund.models.schur_continued import (
    bench_schur_continued,
)
from quant_fund.models.stieltjes_fraction import (
    bench_stieltjes_fraction,
)


def test_pade_approx():
    assert bench_pade_approx()["synthetic_pade_approx"] == 1.0


def test_rational_chebyshev():
    assert bench_rational_chebyshev()["synthetic_rational_chebyshev"] == 1.0


def test_stieltjes_fraction():
    assert bench_stieltjes_fraction()["synthetic_stieltjes_fraction"] == 1.0


def test_loewner_interp():
    assert bench_loewner_interp()["synthetic_loewner_interp"] == 1.0


def test_nevanlinna_pick():
    assert bench_nevanlinna_pick()["synthetic_nevanlinna_pick"] == 1.0


def test_schur_continued():
    assert bench_schur_continued()["synthetic_schur_continued"] == 1.0
