"""Wave-1012 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1012 import (
    bench_bcs_theory_family,
    bench_cabibbo_km_family,
    bench_nuclear_liquid_drop_family,
    bench_nuclear_shell_model_family,
    bench_parton_model_family,
    bench_quark_model_family,
)

_FAMILY_BENCHES = [
    bench_bcs_theory_family,
    bench_nuclear_shell_model_family,
    bench_nuclear_liquid_drop_family,
    bench_quark_model_family,
    bench_parton_model_family,
    bench_cabibbo_km_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
