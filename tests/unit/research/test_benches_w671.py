"""Wave-671 homotopy-24 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w671 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gray_periodic": b.bench_gray_periodic_family(),
        "stunted_proj": b.bench_stunted_proj_family(),
        "adams_edge": b.bench_adams_edge_family(),
        "periodic_family": b.bench_periodic_family_family(),
        "unstable_adams2": b.bench_unstable_adams2_family(),
        "homotopy_exponent": b.bench_homotopy_exponent_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
