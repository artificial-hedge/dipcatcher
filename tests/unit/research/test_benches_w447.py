"""Wave-447 TQFT adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w447 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tqft_axiom": b.bench_tqft_axiom_family(),
        "bord_cat": b.bench_bord_cat_family(),
        "frobenius_2d": b.bench_frobenius_2d_family(),
        "extended_tqft": b.bench_extended_tqft_family(),
        "dw_theory": b.bench_dw_theory_family(),
        "chern_simons": b.bench_chern_simons_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
