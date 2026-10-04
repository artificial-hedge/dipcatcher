from quant_fund.models.borel_tanner import bench_borel_tanner
from quant_fund.models.engset import bench_engset
from quant_fund.models.erlang_b import bench_erlang_b
from quant_fund.models.erlang_c import bench_erlang_c
from quant_fund.models.pollaczek_khinchine import (
    bench_pollaczek_khinchine,
)
from quant_fund.models.takacs_vacation import (
    bench_takacs_vacation,
)


def test_engset():
    assert bench_engset()["synthetic_engset"] == 1.0


def test_erlang_b():
    assert bench_erlang_b()["synthetic_erlang_b"] == 1.0


def test_erlang_c():
    assert bench_erlang_c()["synthetic_erlang_c"] == 1.0


def test_pollaczek_khinchine():
    assert bench_pollaczek_khinchine()["synthetic_pollaczek_khinchine"] == 1.0


def test_borel_tanner():
    assert bench_borel_tanner()["synthetic_borel_tanner"] == 1.0


def test_takacs_vacation():
    assert bench_takacs_vacation()["synthetic_takacs_vacation"] == 1.0
