from quant_fund.models.analytic_ring2 import bench_analytic_ring2
from quant_fund.models.clausen_scholze import bench_clausen_scholze
from quant_fund.models.pyknotic import bench_pyknotic
from quant_fund.models.solid_derived import bench_solid_derived
from quant_fund.models.solid_tensor import bench_solid_tensor
from quant_fund.models.trace_class import bench_trace_class


def test_analytic_ring2():
    assert bench_analytic_ring2()["synthetic_analytic_ring2"] == 1.0


def test_solid_tensor():
    assert bench_solid_tensor()["synthetic_solid_tensor"] == 1.0


def test_trace_class():
    assert bench_trace_class()["synthetic_trace_class"] == 1.0


def test_clausen_scholze():
    assert bench_clausen_scholze()["synthetic_clausen_scholze"] == 1.0


def test_solid_derived():
    assert bench_solid_derived()["synthetic_solid_derived"] == 1.0


def test_pyknotic():
    assert bench_pyknotic()["synthetic_pyknotic"] == 1.0
