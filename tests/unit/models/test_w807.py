from quant_fund.models.cayley_moser import (
    bench_cayley_moser,
)
from quant_fund.models.chow_robbins import (
    bench_chow_robbins,
)
from quant_fund.models.free_boundary import (
    bench_free_boundary,
)
from quant_fund.models.markov_stopping import (
    bench_markov_stopping,
)
from quant_fund.models.secretary_dp import (
    bench_secretary_dp,
)
from quant_fund.models.snell_envelope import (
    bench_snell_envelope,
)


def test_snell_envelope():
    assert bench_snell_envelope()["synthetic_snell_envelope"] == 1.0


def test_secretary_dp():
    assert bench_secretary_dp()["synthetic_secretary_dp"] == 1.0


def test_cayley_moser():
    assert bench_cayley_moser()["synthetic_cayley_moser"] == 1.0


def test_chow_robbins():
    assert bench_chow_robbins()["synthetic_chow_robbins"] == 1.0


def test_markov_stopping():
    assert bench_markov_stopping()["synthetic_markov_stopping"] == 1.0


def test_free_boundary():
    assert bench_free_boundary()["synthetic_free_boundary"] == 1.0
