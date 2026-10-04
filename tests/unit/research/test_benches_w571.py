"""Wave-571 singularity-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w571 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "du_val_sing": b.bench_du_val_sing_family(),
        "rational_sing": b.bench_rational_sing_family(),
        "log_canonical": b.bench_log_canonical_family(),
        "multiplier_ideal": b.bench_multiplier_ideal_family(),
        "bernstein_sato": b.bench_bernstein_sato_family(),
        "milnor_fiber": b.bench_milnor_fiber_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
