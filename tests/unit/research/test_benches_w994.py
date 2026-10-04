"""Wave-994 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w994 import (
    bench_borg_levinson_family,
    bench_gelfand_levitan_family,
    bench_inverse_scattering_family,
    bench_kdv_isospectral_family,
    bench_marchenko_eq_family,
    bench_trace_formulas_family,
)

_FAMILY_BENCHES = [
    bench_inverse_scattering_family,
    bench_marchenko_eq_family,
    bench_gelfand_levitan_family,
    bench_kdv_isospectral_family,
    bench_trace_formulas_family,
    bench_borg_levinson_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
