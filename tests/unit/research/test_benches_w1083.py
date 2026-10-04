"""Wave-1083 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1083 import (
    bench_hermeneutics_family,
    bench_narratology_family,
    bench_phenomenology_family,
    bench_poststructuralism_family,
    bench_semiotics_family,
    bench_structuralism_family,
)

_FAMILY_BENCHES = [
    bench_semiotics_family,
    bench_narratology_family,
    bench_hermeneutics_family,
    bench_phenomenology_family,
    bench_structuralism_family,
    bench_poststructuralism_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
