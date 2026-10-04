"""Wave-798 forward-SDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w798 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "forward_sde": b.bench_forward_sde_family(),
        "random_sde": b.bench_random_sde_family(),
        "anticipating_sde": b.bench_anticipating_sde_family(),
        "functional_sde": b.bench_functional_sde_family(),
        "delayed_sde": b.bench_delayed_sde_family(),
        "neutral_sde": b.bench_neutral_sde_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
