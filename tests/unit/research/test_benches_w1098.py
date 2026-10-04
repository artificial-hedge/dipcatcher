"""Wave-1098 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1098 import (
    bench_biological_anthropology_family,
    bench_economic_anthropology_family,
    bench_medical_anthropology_family,
    bench_paleoanthropology_family,
    bench_political_anthropology_family,
    bench_urban_anthropology_family,
)

_FAMILY_BENCHES = [
    bench_biological_anthropology_family,
    bench_paleoanthropology_family,
    bench_medical_anthropology_family,
    bench_economic_anthropology_family,
    bench_political_anthropology_family,
    bench_urban_anthropology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
