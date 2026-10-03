"""Wave-538 elliptic-PDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w538 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sobolev_space": b.bench_sobolev_space_family(),
        "poincare_ineq": b.bench_poincare_ineq_family(),
        "trace_thm": b.bench_trace_thm_family(),
        "harnack_thm": b.bench_harnack_thm_family(),
        "schauder_est": b.bench_schauder_est_family(),
        "degiorgi_nash": b.bench_degiorgi_nash_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
