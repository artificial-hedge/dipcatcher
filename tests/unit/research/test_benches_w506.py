"""Wave-506 NIP/distal model-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w506 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dp_rank": b.bench_dp_rank_family(),
        "forking_seq": b.bench_forking_seq_family(),
        "honest_def": b.bench_honest_def_family(),
        "uniform_def": b.bench_uniform_def_family(),
        "distality": b.bench_distality_family(),
        "nip_formula": b.bench_nip_formula_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
