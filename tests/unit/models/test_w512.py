from quant_fund.models.berezin_int import bench_berezin_int
from quant_fund.models.odd_variables import bench_odd_variables
from quant_fund.models.super_lie import bench_super_lie
from quant_fund.models.super_manifold import bench_super_manifold
from quant_fund.models.super_scheme import bench_super_scheme
from quant_fund.models.super_space import bench_super_space


def test_super_space():
    assert bench_super_space()["synthetic_super_space"] == 1.0


def test_super_manifold():
    assert bench_super_manifold()["synthetic_super_manifold"] == 1.0


def test_super_lie():
    assert bench_super_lie()["synthetic_super_lie"] == 1.0


def test_odd_variables():
    assert bench_odd_variables()["synthetic_odd_variables"] == 1.0


def test_berezin_int():
    assert bench_berezin_int()["synthetic_berezin_int"] == 1.0


def test_super_scheme():
    assert bench_super_scheme()["synthetic_super_scheme"] == 1.0
