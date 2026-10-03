"""Wave-345 commutative-algebra adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w345 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ring_ideals": b.bench_ring_ideals_family(),
        "quotient_ring": b.bench_quotient_ring_family(),
        "pid_check": b.bench_pid_check_family(),
        "minimal_poly": b.bench_minimal_poly_family(),
        "norm_trace": b.bench_norm_trace_family(),
        "spec_ring": b.bench_spec_ring_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
