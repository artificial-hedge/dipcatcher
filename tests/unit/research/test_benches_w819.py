"""Wave-819 stopping-time adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w819 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "first_hitting": b.bench_first_hitting_family(),
        "last_exit": b.bench_last_exit_family(),
        "stopping_sigma": b.bench_stopping_sigma_family(),
        "progressive_set": b.bench_progressive_set_family(),
        "debuts_theorem": b.bench_debuts_theorem_family(),
        "accessible_time": b.bench_accessible_time_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
