"""Wave-1080 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1080 import (
    bench_byzantine_studies_family,
    bench_codicology_family,
    bench_hagiography_family,
    bench_medieval_studies_family,
    bench_numismatics_family,
    bench_paleography_family,
)

_FAMILY_BENCHES = [
    bench_medieval_studies_family,
    bench_paleography_family,
    bench_codicology_family,
    bench_hagiography_family,
    bench_byzantine_studies_family,
    bench_numismatics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
