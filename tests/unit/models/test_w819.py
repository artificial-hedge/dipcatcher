from quant_fund.models.accessible_time import (
    bench_accessible_time,
)
from quant_fund.models.debuts_theorem import (
    bench_debuts_theorem,
)
from quant_fund.models.first_hitting import (
    bench_first_hitting,
)
from quant_fund.models.last_exit import (
    bench_last_exit,
)
from quant_fund.models.progressive_set import (
    bench_progressive_set,
)
from quant_fund.models.stopping_sigma import (
    bench_stopping_sigma,
)


def test_first_hitting():
    assert bench_first_hitting()["synthetic_first_hitting"] == 1.0


def test_last_exit():
    assert bench_last_exit()["synthetic_last_exit"] == 1.0


def test_stopping_sigma():
    assert bench_stopping_sigma()["synthetic_stopping_sigma"] == 1.0


def test_progressive_set():
    assert bench_progressive_set()["synthetic_progressive_set"] == 1.0


def test_debuts_theorem():
    assert bench_debuts_theorem()["synthetic_debuts_theorem"] == 1.0


def test_accessible_time():
    assert bench_accessible_time()["synthetic_accessible_time"] == 1.0
