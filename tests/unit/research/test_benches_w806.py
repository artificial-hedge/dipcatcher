"""Wave-806 jump-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w806 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "jump_diffusion": b.bench_jump_diffusion_family(),
        "merton_jump": b.bench_merton_jump_family(),
        "kou_model": b.bench_kou_model_family(),
        "compound_poisson": b.bench_compound_poisson_family(),
        "excursion_theory": b.bench_excursion_theory_family(),
        "marked_hawkes": b.bench_marked_hawkes_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
