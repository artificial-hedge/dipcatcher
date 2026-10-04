"""Wave-958 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w958 import (
    bench_araki_lieb_thirring_family,
    bench_hadamard_fischer_family,
    bench_ky_fan_family,
    bench_lidskii_thm_family,
    bench_pinching_ineq_family,
    bench_von_neumann_trace_family,
)

_FAMILY_BENCHES = [
    bench_ky_fan_family,
    bench_lidskii_thm_family,
    bench_von_neumann_trace_family,
    bench_pinching_ineq_family,
    bench_araki_lieb_thirring_family,
    bench_hadamard_fischer_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
