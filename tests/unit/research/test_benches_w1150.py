"""Wave-1150 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1150 import (
    bench_bacteriology_family,
    bench_epigenetics_family,
    bench_immunogenetics_family,
    bench_microbiology_2_family,
    bench_molecular_genetics_family,
    bench_virology_2_family,
)

_FAMILY_BENCHES = [
    bench_microbiology_2_family,
    bench_bacteriology_family,
    bench_virology_2_family,
    bench_immunogenetics_family,
    bench_molecular_genetics_family,
    bench_epigenetics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
