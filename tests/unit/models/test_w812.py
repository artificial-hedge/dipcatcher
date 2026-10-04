from quant_fund.models.epsilon_coupling import (
    bench_epsilon_coupling,
)
from quant_fund.models.nummelin import (
    bench_nummelin,
)
from quant_fund.models.petite_set import (
    bench_petite_set,
)
from quant_fund.models.regenerative import (
    bench_regenerative,
)
from quant_fund.models.small_set import (
    bench_small_set,
)
from quant_fund.models.split_chain import (
    bench_split_chain,
)


def test_regenerative():
    assert bench_regenerative()["synthetic_regenerative"] == 1.0


def test_epsilon_coupling():
    assert bench_epsilon_coupling()["synthetic_epsilon_coupling"] == 1.0


def test_small_set():
    assert bench_small_set()["synthetic_small_set"] == 1.0


def test_petite_set():
    assert bench_petite_set()["synthetic_petite_set"] == 1.0


def test_split_chain():
    assert bench_split_chain()["synthetic_split_chain"] == 1.0


def test_nummelin():
    assert bench_nummelin()["synthetic_nummelin"] == 1.0
