"""Wave-628 operads-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w628 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "moerdijk_weiss": b.bench_moerdijk_weiss_family(),
        "higher_operad": b.bench_higher_operad_family(),
        "operad_infty3": b.bench_operad_infty3_family(),
        "operad_cat2": b.bench_operad_cat2_family(),
        "dendroidal_seg": b.bench_dendroidal_seg_family(),
        "operad_module": b.bench_operad_module_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
