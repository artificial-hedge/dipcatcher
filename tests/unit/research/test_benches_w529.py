"""Wave-529 KAM/Aubry-Mather adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w529 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kam_theorem": b.bench_kam_theorem_family(),
        "aubry_mather": b.bench_aubry_mather_family(),
        "twist_map": b.bench_twist_map_family(),
        "cantorus": b.bench_cantorus_family(),
        "greene_crit": b.bench_greene_crit_family(),
        "arnold_diff": b.bench_arnold_diff_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
