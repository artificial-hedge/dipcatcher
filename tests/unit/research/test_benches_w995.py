"""Wave-995 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w995 import (
    bench_calogero_moser_family,
    bench_kp_hierarchy_family,
    bench_nls_soliton_family,
    bench_painleve_eq_family,
    bench_sine_gordon_family,
    bench_toda_lattice_family,
)

_FAMILY_BENCHES = [
    bench_sine_gordon_family,
    bench_nls_soliton_family,
    bench_toda_lattice_family,
    bench_calogero_moser_family,
    bench_kp_hierarchy_family,
    bench_painleve_eq_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
