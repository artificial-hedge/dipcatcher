"""Wave-1158 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1158 import (
    bench_anthropology_6_family,
    bench_economics_6_family,
    bench_linguistics_7_family,
    bench_political_science_3_family,
    bench_psychology_5_family,
    bench_sociology_6_family,
)

_FAMILY_BENCHES = [
    bench_sociology_6_family,
    bench_economics_6_family,
    bench_political_science_3_family,
    bench_psychology_5_family,
    bench_anthropology_6_family,
    bench_linguistics_7_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
