"""Wave-1181 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1181 import (
    bench_acting_studies_family,
    bench_directing_studies_family,
    bench_performing_arts_2_family,
    bench_playwriting_family,
    bench_scenography_family,
    bench_theater_arts_family,
)

_FAMILY_BENCHES = [
    bench_performing_arts_2_family,
    bench_theater_arts_family,
    bench_acting_studies_family,
    bench_directing_studies_family,
    bench_playwriting_family,
    bench_scenography_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
