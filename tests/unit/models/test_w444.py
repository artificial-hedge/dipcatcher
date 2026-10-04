from quant_fund.models.cotangent_cx import bench_cotangent_cx
from quant_fund.models.derived_stack import bench_derived_stack
from quant_fund.models.geometric_stk import bench_geometric_stk
from quant_fund.models.perf_stack import bench_perf_stack
from quant_fund.models.quasi_smooth import bench_quasi_smooth
from quant_fund.models.tannaka_rec import bench_tannaka_rec


def test_derived_stack():
    assert bench_derived_stack()["synthetic_derived_stack"] == 1.0


def test_cotangent_cx():
    assert bench_cotangent_cx()["synthetic_cotangent_cx"] == 1.0


def test_geometric_stk():
    assert bench_geometric_stk()["synthetic_geometric_stk"] == 1.0


def test_tannaka_rec():
    assert bench_tannaka_rec()["synthetic_tannaka_rec"] == 1.0


def test_quasi_smooth():
    assert bench_quasi_smooth()["synthetic_quasi_smooth"] == 1.0


def test_perf_stack():
    assert bench_perf_stack()["synthetic_perf_stack"] == 1.0
