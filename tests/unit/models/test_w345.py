from quant_fund.models.minimal_poly import bench_minimal_poly
from quant_fund.models.norm_trace import bench_norm_trace
from quant_fund.models.pid_check import bench_pid_check
from quant_fund.models.quotient_ring import bench_quotient_ring
from quant_fund.models.ring_ideals import bench_ring_ideals
from quant_fund.models.spec_ring import bench_spec_ring


def test_ring_ideals():
    assert bench_ring_ideals()["synthetic_ring_ideals"] == 1.0


def test_quotient_ring():
    assert bench_quotient_ring()["synthetic_quotient_ring"] == 1.0


def test_pid_check():
    assert bench_pid_check()["synthetic_pid_check"] == 1.0


def test_minimal_poly():
    assert bench_minimal_poly()["synthetic_minimal_poly"] == 1.0


def test_norm_trace():
    assert bench_norm_trace()["synthetic_norm_trace"] == 1.0


def test_spec_ring():
    assert bench_spec_ring()["synthetic_spec_ring"] == 1.0
