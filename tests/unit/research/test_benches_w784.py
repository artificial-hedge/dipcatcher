"""Wave-784 Levy adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w784 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "levy_khinchine": b.bench_levy_khinchine_family(),
        "subordinator": b.bench_subordinator_family(),
        "stable_levy": b.bench_stable_levy_family(),
        "self_decomp": b.bench_self_decomp_family(),
        "levy_measure": b.bench_levy_measure_family(),
        "girsanov_thm2": b.bench_girsanov_thm2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
