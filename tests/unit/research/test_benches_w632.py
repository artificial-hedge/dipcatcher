"""Wave-632 witt-vectors-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w632 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "witt_len2": b.bench_witt_len2_family(),
        "big_witt": b.bench_big_witt_family(),
        "good_reduction": b.bench_good_reduction_family(),
        "potential_reduction": b.bench_potential_reduction_family(),
        "tate_curve": b.bench_tate_curve_family(),
        "odeur_zarba": b.bench_odeur_zarba_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
