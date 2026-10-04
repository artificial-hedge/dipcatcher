"""Wave-1085 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1085 import (
    bench_medieval_literature_family,
    bench_modernism_family,
    bench_postmodernism_family,
    bench_renaissance_literature_family,
    bench_romanticism_family,
    bench_victorian_studies_family,
)

_FAMILY_BENCHES = [
    bench_medieval_literature_family,
    bench_renaissance_literature_family,
    bench_romanticism_family,
    bench_modernism_family,
    bench_postmodernism_family,
    bench_victorian_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
