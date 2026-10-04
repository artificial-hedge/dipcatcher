"""Wave-1002 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1002 import (
    bench_energy_spectrum_family,
    bench_intermittency_models_family,
    bench_kolmogorov_theory_family,
    bench_reynolds_decomp_family,
    bench_taylor_series_hyp_family,
    bench_wall_turbulence_family,
)

_FAMILY_BENCHES = [
    bench_kolmogorov_theory_family,
    bench_reynolds_decomp_family,
    bench_energy_spectrum_family,
    bench_intermittency_models_family,
    bench_wall_turbulence_family,
    bench_taylor_series_hyp_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
