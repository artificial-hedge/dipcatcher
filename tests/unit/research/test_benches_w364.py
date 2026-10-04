"""Wave-364 functional-analysis-3 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w364 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hahn_banach": b.bench_hahn_banach_family(),
        "riesz_repr": b.bench_riesz_repr_family(),
        "adjoint_op": b.bench_adjoint_op_family(),
        "selfadjoint_spectrum": b.bench_selfadjoint_spectrum_family(),
        "compact_resolvent": b.bench_compact_resolvent_family(),
        "projection_thm": b.bench_projection_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
