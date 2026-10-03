"""Wave-326 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w326 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "timed_automata": b.bench_timed_automata_family(),
        "parity_game": b.bench_parity_game_family(),
        "nba_emptiness": b.bench_nba_emptiness_family(),
        "ctl_mc": b.bench_ctl_mc_family(),
        "bisim_refine": b.bench_bisim_refine_family(),
        "wsts_cover": b.bench_wsts_cover_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
