"""Wave-544 Kleinian-groups adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w544 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kleinian_group": b.bench_kleinian_group_family(),
        "limit_set": b.bench_limit_set_family(),
        "hyperbolic_3mfd": b.bench_hyperbolic_3mfd_family(),
        "mostow_rigidity": b.bench_mostow_rigidity_family(),
        "jorgensen_thurston": b.bench_jorgensen_thurston_family(),
        "tameness_thm": b.bench_tameness_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
