"""Wave-479 p-adic-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w479 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fargues_diam": b.bench_fargues_diam_family(),
        "tilting_equiv": b.bench_tilting_equiv_family(),
        "scholze_diamond": b.bench_scholze_diamond_family(),
        "ahb_ring": b.bench_ahb_ring_family(),
        "prism_2": b.bench_prism_2_family(),
        "drinfeld_sym": b.bench_drinfeld_sym_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
