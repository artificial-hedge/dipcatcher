"""Wave-1006 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1006 import (
    bench_fock_space_family,
    bench_harmonic_oscillator_family,
    bench_hydrogen_atom_family,
    bench_schrodinger_eq_family,
    bench_spin_half_family,
    bench_wigner_wick_family,
)

_FAMILY_BENCHES = [
    bench_schrodinger_eq_family,
    bench_hydrogen_atom_family,
    bench_harmonic_oscillator_family,
    bench_spin_half_family,
    bench_wigner_wick_family,
    bench_fock_space_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
