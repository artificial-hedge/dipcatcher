"""Wave-559 complex-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w559 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "calabi_yau_mfd": b.bench_calabi_yau_mfd_family(),
        "calabi_conjecture": b.bench_calabi_conjecture_family(),
        "kahler_einstein": b.bench_kahler_einstein_family(),
        "k_stability": b.bench_k_stability_family(),
        "csck_metric": b.bench_csck_metric_family(),
        "futaki_invariant": b.bench_futaki_invariant_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
