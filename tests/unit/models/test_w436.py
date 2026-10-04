from quant_fund.models.adjoint_functor import (
    bench_adjoint_functor,
)
from quant_fund.models.bousfield_loc import bench_bousfield_loc
from quant_fund.models.cartesian_fib import bench_cartesian_fib
from quant_fund.models.complete_seg import bench_complete_seg
from quant_fund.models.presentable_cat import bench_presentable_cat
from quant_fund.models.straightening import bench_straightening


def test_complete_seg():
    assert bench_complete_seg()["synthetic_complete_seg"] == 1.0


def test_cartesian_fib():
    assert bench_cartesian_fib()["synthetic_cartesian_fib"] == 1.0


def test_straightening():
    assert bench_straightening()["synthetic_straightening"] == 1.0


def test_presentable_cat():
    assert bench_presentable_cat()["synthetic_presentable_cat"] == 1.0


def test_adjoint_functor():
    assert bench_adjoint_functor()["synthetic_adjoint_functor"] == 1.0


def test_bousfield_loc():
    assert bench_bousfield_loc()["synthetic_bousfield_loc"] == 1.0
