from quant_fund.models.doss_sussmann import (
    bench_doss_sussmann,
)
from quant_fund.models.follmer_strat import (
    bench_follmer_strat,
)
from quant_fund.models.ito_isometry import (
    bench_ito_isometry,
)
from quant_fund.models.skorohod_lemma import (
    bench_skorohod_lemma,
)
from quant_fund.models.stratonovich_conv import (
    bench_stratonovich_conv,
)
from quant_fund.models.tanaka_meyer import (
    bench_tanaka_meyer,
)


def test_ito_isometry():
    assert bench_ito_isometry()["synthetic_ito_isometry"] == 1.0


def test_stratonovich_conv():
    assert bench_stratonovich_conv()["synthetic_stratonovich_conv"] == 1.0


def test_tanaka_meyer():
    assert bench_tanaka_meyer()["synthetic_tanaka_meyer"] == 1.0


def test_follmer_strat():
    assert bench_follmer_strat()["synthetic_follmer_strat"] == 1.0


def test_skorohod_lemma():
    assert bench_skorohod_lemma()["synthetic_skorohod_lemma"] == 1.0


def test_doss_sussmann():
    assert bench_doss_sussmann()["synthetic_doss_sussmann"] == 1.0
