"""Wave-795 stochastic-games adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w795 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dynkin_game": b.bench_dynkin_game_family(),
        "stochastic_game2": b.bench_stochastic_game2_family(),
        "differential_game": b.bench_differential_game_family(),
        "zero_sum_game": b.bench_zero_sum_game_family(),
        "nonzero_sum_game": b.bench_nonzero_sum_game_family(),
        "isaacs_equation": b.bench_isaacs_equation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
