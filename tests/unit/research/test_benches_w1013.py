"""Wave-1013 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1013 import (
    bench_born_oppenheimer_family,
    bench_hartree_fock_family,
    bench_molecular_orbitals_family,
    bench_rotational_spectra_family,
    bench_vibrational_spectra_family,
    bench_zeeman_effect_family,
)

_FAMILY_BENCHES = [
    bench_hartree_fock_family,
    bench_born_oppenheimer_family,
    bench_molecular_orbitals_family,
    bench_rotational_spectra_family,
    bench_vibrational_spectra_family,
    bench_zeeman_effect_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
