"""Wave-808 filtering adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w808 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "zakai_eq": b.bench_zakai_eq_family(),
        "kushner_strat": b.bench_kushner_strat_family(),
        "kalman_bucy": b.bench_kalman_bucy_family(),
        "bene_filter": b.bench_bene_filter_family(),
        "hidden_markov_filter": b.bench_hidden_markov_filter_family(),
        "particle_filter2": b.bench_particle_filter2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
