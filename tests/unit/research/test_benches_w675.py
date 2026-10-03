"""Wave-675 higher-algebra-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w675 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e3_algebra": b.bench_e3_algebra_family(),
        "getzler_jones": b.bench_getzler_jones_family(),
        "tadv_hochschild": b.bench_tadv_hochschild_family(),
        "cyclotomic_e_n": b.bench_cyclotomic_e_n_family(),
        "surfaces_operad": b.bench_surfaces_operad_family(),
        "boards_operad": b.bench_boards_operad_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
