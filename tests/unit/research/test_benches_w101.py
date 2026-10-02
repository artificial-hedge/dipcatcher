import numpy as np
import pytest

from quant_fund.research.benches_w101 import (
    bench_assignment_family,
    bench_exact_cover_family,
    bench_graph_components_family,
    bench_graph_traversal_family,
    bench_network_flow_family,
    bench_shortest_paths_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES = {
    "graph_traversal": bench_graph_traversal_family,
    "shortest_paths": bench_shortest_paths_family,
    "network_flow": bench_network_flow_family,
    "assignment": bench_assignment_family,
    "graph_components": bench_graph_components_family,
    "exact_cover": bench_exact_cover_family,
}


def test_w101_families_registered():
    for name in _FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES, name


@pytest.mark.parametrize("name,fn", list(_FAMILIES.items()))
def test_w101_bench_outputs_finite(name, fn):
    out = fn()
    assert isinstance(out, dict) and len(out) > 0
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v), (name, k, v)
