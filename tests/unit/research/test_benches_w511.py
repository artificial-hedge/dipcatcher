"""Wave-511 Fargues-Scholze adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w511 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fs_diamond": b.bench_fs_diamond_family(),
        "geometric_satake": b.bench_geometric_satake_family(),
        "v_sheaf": b.bench_v_sheaf_family(),
        "bun_g": b.bench_bun_g_family(),
        "hecke_stack": b.bench_hecke_stack_family(),
        "y_diamond": b.bench_y_diamond_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
