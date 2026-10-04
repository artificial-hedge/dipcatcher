"""Wave-1043 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1043 import (
    bench_dendrology_family,
    bench_forest_ecology_family,
    bench_forest_economics_family,
    bench_silviculture_family,
    bench_timber_harvesting_family,
    bench_wildfire_management_family,
)

_FAMILY_BENCHES = [
    bench_silviculture_family,
    bench_forest_ecology_family,
    bench_timber_harvesting_family,
    bench_forest_economics_family,
    bench_dendrology_family,
    bench_wildfire_management_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
