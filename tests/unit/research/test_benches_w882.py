"""Wave-882 preconditioner/DD adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w882 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spai_precond": b.bench_spai_precond_family(),
        "diagonal_scale": b.bench_diagonal_scale_family(),
        "nonoverlap_dd": b.bench_nonoverlap_dd_family(),
        "overlap_dd": b.bench_overlap_dd_family(),
        "restrictive_dd": b.bench_restrictive_dd_family(),
        "balanced_dd": b.bench_balanced_dd_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
