"""Wave-1142 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1142 import (
    bench_quantum_chemistry_2_family,
    bench_quantum_computing_family,
    bench_quantum_error_2_family,
    bench_quantum_information_2_family,
    bench_quantum_optics_family,
    bench_quantum_sensing_family,
)

_FAMILY_BENCHES = [
    bench_quantum_computing_family,
    bench_quantum_information_2_family,
    bench_quantum_chemistry_2_family,
    bench_quantum_optics_family,
    bench_quantum_sensing_family,
    bench_quantum_error_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
