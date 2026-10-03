from quant_fund.models.completion_ring import bench_completion_ring
from quant_fund.models.dimension_fiber import bench_dimension_fiber
from quant_fund.models.hilbert_samuel import bench_hilbert_samuel
from quant_fund.models.krull_dim import bench_krull_dim
from quant_fund.models.noether_normal import bench_noether_normal
from quant_fund.models.primary_decomp import bench_primary_decomp


def test_hilbert_samuel():
    assert bench_hilbert_samuel()["synthetic_hilbert_samuel"] == 1.0


def test_krull_dim():
    assert bench_krull_dim()["synthetic_krull_dim"] == 1.0


def test_noether_normal():
    assert bench_noether_normal()["synthetic_noether_normal"] == 1.0


def test_primary_decomp():
    assert bench_primary_decomp()["synthetic_primary_decomp"] == 1.0


def test_completion_ring():
    assert bench_completion_ring()["synthetic_completion_ring"] == 1.0


def test_dimension_fiber():
    assert bench_dimension_fiber()["synthetic_dimension_fiber"] == 1.0
