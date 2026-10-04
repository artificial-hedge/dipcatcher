from quant_fund.models.absolute_hodge import bench_absolute_hodge
from quant_fund.models.griffiths_transv import (
    bench_griffiths_transv,
)
from quant_fund.models.hodge_class import bench_hodge_class
from quant_fund.models.hodge_conj import bench_hodge_conj
from quant_fund.models.mumford_tate import bench_mumford_tate
from quant_fund.models.period_domain import bench_period_domain


def test_griffiths_transv():
    assert bench_griffiths_transv()["synthetic_griffiths_transv"] == 1.0


def test_period_domain():
    assert bench_period_domain()["synthetic_period_domain"] == 1.0


def test_mumford_tate():
    assert bench_mumford_tate()["synthetic_mumford_tate"] == 1.0


def test_hodge_class():
    assert bench_hodge_class()["synthetic_hodge_class"] == 1.0


def test_absolute_hodge():
    assert bench_absolute_hodge()["synthetic_absolute_hodge"] == 1.0


def test_hodge_conj():
    assert bench_hodge_conj()["synthetic_hodge_conj"] == 1.0
