"""Wave-741 percolation-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w741 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "beffara_nolin": b.bench_beffara_nolin_family(),
        "hara_slade": b.bench_hara_slade_family(),
        "gandre_liggett": b.bench_gandre_liggett_family(),
        "heyman_redner": b.bench_heyman_redner_family(),
        "aiten_chayes": b.bench_aiten_chayes_family(),
        "newman_percolation": b.bench_newman_percolation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
