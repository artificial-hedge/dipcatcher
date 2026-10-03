"""Wave-789 optimal-transport adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w789 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wasserstein_grad": b.bench_wasserstein_grad_family(),
        "jko_step": b.bench_jko_step_family(),
        "benamou_brenier": b.bench_benamou_brenier_family(),
        "entropy_regular": b.bench_entropy_regular_family(),
        "fokker_planck2": b.bench_fokker_planck2_family(),
        "gradient_flow": b.bench_gradient_flow_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
