from quant_fund.models.dirichlet_form import (
    bench_dirichlet_form,
)
from quant_fund.models.hypercontractive import (
    bench_hypercontractive,
)
from quant_fund.models.log_sobolev_sem import (
    bench_log_sobolev_sem,
)
from quant_fund.models.markov_semigroup import (
    bench_markov_semigroup,
)
from quant_fund.models.poincare_semigroup import (
    bench_poincare_semigroup,
)
from quant_fund.models.spectral_gap_sem import (
    bench_spectral_gap_sem,
)


def test_dirichlet_form():
    assert bench_dirichlet_form()["synthetic_dirichlet_form"] == 1.0


def test_markov_semigroup():
    assert bench_markov_semigroup()["synthetic_markov_semigroup"] == 1.0


def test_poincare_semigroup():
    assert bench_poincare_semigroup()["synthetic_poincare_semigroup"] == 1.0


def test_log_sobolev_sem():
    assert bench_log_sobolev_sem()["synthetic_log_sobolev_sem"] == 1.0


def test_hypercontractive():
    assert bench_hypercontractive()["synthetic_hypercontractive"] == 1.0


def test_spectral_gap_sem():
    assert bench_spectral_gap_sem()["synthetic_spectral_gap_sem"] == 1.0
