"""Wave-1001 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1001 import (
    bench_beale_kato_majda_family,
    bench_euler_equations_family,
    bench_ladyzhenskaya_weak_family,
    bench_leray_theory_family,
    bench_navier_stokes_family,
    bench_vorticity_form_family,
)

_FAMILY_BENCHES = [
    bench_euler_equations_family,
    bench_navier_stokes_family,
    bench_vorticity_form_family,
    bench_beale_kato_majda_family,
    bench_ladyzhenskaya_weak_family,
    bench_leray_theory_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
