"""Wave-647 homotopy-19 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w647 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "selick_htpy": b.bench_selick_htpy_family(),
        "arkowitz_htpy": b.bench_arkowitz_htpy_family(),
        "lin_htpy": b.bench_lin_htpy_family(),
        "kahn_priddy": b.bench_kahn_priddy_family(),
        "bochner_htpy": b.bench_bochner_htpy_family(),
        "tits_building": b.bench_tits_building_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
