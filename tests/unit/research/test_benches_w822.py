"""Wave-822 filtration adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w822 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "natural_filtration": b.bench_natural_filtration_family(),
        "right_continuous_f": b.bench_right_continuous_f_family(),
        "usual_aug": b.bench_usual_aug_family(),
        "enlargement_f": b.bench_enlargement_f_family(),
        "initial_enlarg": b.bench_initial_enlarg_family(),
        "progressive_enlarg": b.bench_progressive_enlarg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
