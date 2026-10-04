from quant_fund.models.derived_critical import bench_derived_critical
from quant_fund.models.lagrangian_int import bench_lagrangian_int
from quant_fund.models.lie_algebroid import bench_lie_algebroid
from quant_fund.models.moment_map import bench_moment_map
from quant_fund.models.quant_dag import bench_quant_dag
from quant_fund.models.shifted_sympl import bench_shifted_sympl


def test_shifted_sympl():
    assert bench_shifted_sympl()["synthetic_shifted_sympl"] == 1.0


def test_lagrangian_int():
    assert bench_lagrangian_int()["synthetic_lagrangian_int"] == 1.0


def test_derived_critical():
    assert bench_derived_critical()["synthetic_derived_critical"] == 1.0


def test_lie_algebroid():
    assert bench_lie_algebroid()["synthetic_lie_algebroid"] == 1.0


def test_moment_map():
    assert bench_moment_map()["synthetic_moment_map"] == 1.0


def test_quant_dag():
    assert bench_quant_dag()["synthetic_quant_dag"] == 1.0
