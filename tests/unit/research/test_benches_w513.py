"""Wave-513 categorification adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w513 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "categorify": b.bench_categorify_family(),
        "khovanov_hom": b.bench_khovanov_hom_family(),
        "hecke_cat": b.bench_hecke_cat_family(),
        "soergel_bim": b.bench_soergel_bim_family(),
        "rasmussen_inv": b.bench_rasmussen_inv_family(),
        "uq_sl2": b.bench_uq_sl2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
