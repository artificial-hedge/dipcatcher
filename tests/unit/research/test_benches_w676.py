"""Wave-676 higher-algebra-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w676 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "en_algebra2": b.bench_en_algebra2_family(),
        "thom_transpose": b.bench_thom_transpose_family(),
        "higher_brace2": b.bench_higher_brace2_family(),
        "koszul_operad2": b.bench_koszul_operad2_family(),
        "operad_lie": b.bench_operad_lie_family(),
        "center_hochschild": b.bench_center_hochschild_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
