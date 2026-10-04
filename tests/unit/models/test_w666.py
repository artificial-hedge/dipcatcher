from quant_fund.models.cotangent_stack import (
    bench_cotangent_stack,
)
from quant_fund.models.derived_abelian import (
    bench_derived_abelian,
)
from quant_fund.models.derived_bezout import (
    bench_derived_bezout,
)
from quant_fund.models.derived_bun import bench_derived_bun
from quant_fund.models.derived_hecke import bench_derived_hecke
from quant_fund.models.simplicial_comm import (
    bench_simplicial_comm,
)


def test_derived_abelian():
    assert bench_derived_abelian()["synthetic_derived_abelian"] == 1.0


def test_simplicial_comm():
    assert bench_simplicial_comm()["synthetic_simplicial_comm"] == 1.0


def test_derived_bezout():
    assert bench_derived_bezout()["synthetic_derived_bezout"] == 1.0


def test_derived_hecke():
    assert bench_derived_hecke()["synthetic_derived_hecke"] == 1.0


def test_cotangent_stack():
    assert bench_cotangent_stack()["synthetic_cotangent_stack"] == 1.0


def test_derived_bun():
    assert bench_derived_bun()["synthetic_derived_bun"] == 1.0
