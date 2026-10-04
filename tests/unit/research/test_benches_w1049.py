"""Wave-1049 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1049 import (
    bench_dental_anatomy_family,
    bench_endodontics_family,
    bench_oral_pathology_family,
    bench_orthodontics_family,
    bench_periodontology_family,
    bench_prosthodontics_family,
)

_FAMILY_BENCHES = [
    bench_dental_anatomy_family,
    bench_oral_pathology_family,
    bench_periodontology_family,
    bench_endodontics_family,
    bench_orthodontics_family,
    bench_prosthodontics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
