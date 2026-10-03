from quant_fund.models.gambler_ruin import bench_gambler_ruin
from quant_fund.models.markov_chain import bench_markov_chain
from quant_fund.models.markov_hitting import bench_markov_hitting
from quant_fund.models.martingale_check import bench_martingale_check
from quant_fund.models.poisson_process import bench_poisson_process
from quant_fund.models.stopping_time import bench_stopping_time


def test_markov_chain():
    assert bench_markov_chain()["synthetic_markov_chain"] == 1.0


def test_martingale_check():
    assert bench_martingale_check()["synthetic_martingale_check"] == 1.0


def test_poisson_process():
    assert bench_poisson_process()["synthetic_poisson_process"] == 1.0


def test_gambler_ruin():
    assert bench_gambler_ruin()["synthetic_gambler_ruin"] == 1.0


def test_stopping_time():
    assert bench_stopping_time()["synthetic_stopping_time"] == 1.0


def test_markov_hitting():
    assert bench_markov_hitting()["synthetic_markov_hitting"] == 1.0
