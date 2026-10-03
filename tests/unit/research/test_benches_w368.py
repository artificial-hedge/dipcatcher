"""Wave-368 model-theory-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w368 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "unification_fol": b.bench_unification_fol_family(),
        "skolem_normal": b.bench_skolem_normal_family(),
        "herbrand_model": b.bench_herbrand_model_family(),
        "presburger": b.bench_presburger_family(),
        "los_theorem": b.bench_los_theorem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
