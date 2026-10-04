"""Wave-1111 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1111 import (
    bench_medicinal_chemistry_family,
    bench_photochemistry_family,
    bench_quantum_chemistry_family,
    bench_spectroscopy_family,
    bench_stereochemistry_family,
    bench_supramolecular_chemistry_family,
)

_FAMILY_BENCHES = [
    bench_quantum_chemistry_family,
    bench_spectroscopy_family,
    bench_photochemistry_family,
    bench_stereochemistry_family,
    bench_supramolecular_chemistry_family,
    bench_medicinal_chemistry_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
