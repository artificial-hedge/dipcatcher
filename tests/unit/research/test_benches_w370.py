"""Wave-370 graph-theory-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w370 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tutte_berge": b.bench_tutte_berge_family(),
        "dirac_ore": b.bench_dirac_ore_family(),
        "turan_theorem": b.bench_turan_theorem_family(),
        "planar_five": b.bench_planar_five_family(),
        "graph_minor": b.bench_graph_minor_family(),
        "ramsey_num": b.bench_ramsey_num_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
