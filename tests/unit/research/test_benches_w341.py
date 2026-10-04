"""Wave-341 modal-logic/topology-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w341 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kripke_semantics": b.bench_kripke_semantics_family(),
        "bisimulation": b.bench_bisimulation_family(),
        "ef_game": b.bench_ef_game_family(),
        "fundamental_group": b.bench_fundamental_group_family(),
        "covering_space": b.bench_covering_space_family(),
        "topo_separation": b.bench_topo_separation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
