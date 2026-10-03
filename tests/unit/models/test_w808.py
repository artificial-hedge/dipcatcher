from quant_fund.models.bene_filter import (
    bench_bene_filter,
)
from quant_fund.models.hidden_markov_filter import (
    bench_hidden_markov_filter,
)
from quant_fund.models.kalman_bucy import (
    bench_kalman_bucy,
)
from quant_fund.models.kushner_strat import (
    bench_kushner_strat,
)
from quant_fund.models.particle_filter2 import (
    bench_particle_filter2,
)
from quant_fund.models.zakai_eq import (
    bench_zakai_eq,
)


def test_zakai_eq():
    assert bench_zakai_eq()["synthetic_zakai_eq"] == 1.0


def test_kushner_strat():
    assert bench_kushner_strat()["synthetic_kushner_strat"] == 1.0


def test_kalman_bucy():
    assert bench_kalman_bucy()["synthetic_kalman_bucy"] == 1.0


def test_bene_filter():
    assert bench_bene_filter()["synthetic_bene_filter"] == 1.0


def test_hidden_markov_filter():
    assert bench_hidden_markov_filter()["synthetic_hidden_markov_filter"] == 1.0


def test_particle_filter2():
    assert bench_particle_filter2()["synthetic_particle_filter2"] == 1.0
