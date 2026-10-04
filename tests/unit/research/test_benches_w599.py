"""Wave-599 monad-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w599 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "monad_theorem": b.bench_monad_theorem_family(),
        "klesli_cat": b.bench_klesli_cat_family(),
        "codensity_monad": b.bench_codensity_monad_family(),
        "monadicity": b.bench_monadicity_family(),
        "distributive_law": b.bench_distributive_law_family(),
        "algebra_cat": b.bench_algebra_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
