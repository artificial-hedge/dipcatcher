"""Wave-402 topos-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w402 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "topos_subobj": b.bench_topos_subobj_family(),
        "groth_topo": b.bench_groth_topo_family(),
        "sheaf_cond": b.bench_sheaf_cond_family(),
        "logic_topos": b.bench_logic_topos_family(),
        "geometric_morph": b.bench_geometric_morph_family(),
        "etale_space": b.bench_etale_space_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
