"""Wave-977 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w977 import (
    bench_bochner_integral_family,
    bench_bochner_meas_family,
    bench_lusin_rep_family,
    bench_norm_integrable_family,
    bench_pettis_weak_family,
    bench_radon_nikodym_prop_family,
)

_FAMILY_BENCHES = [
    bench_bochner_integral_family,
    bench_lusin_rep_family,
    bench_radon_nikodym_prop_family,
    bench_bochner_meas_family,
    bench_norm_integrable_family,
    bench_pettis_weak_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
