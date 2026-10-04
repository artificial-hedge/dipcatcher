"""Wave-501 F-singularity adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w501 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "f_regular": b.bench_f_regular_family(),
        "f_rational": b.bench_f_rational_family(),
        "f_pure": b.bench_f_pure_family(),
        "f_threshold": b.bench_f_threshold_family(),
        "test_ideal": b.bench_test_ideal_family(),
        "tight_closure": b.bench_tight_closure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
