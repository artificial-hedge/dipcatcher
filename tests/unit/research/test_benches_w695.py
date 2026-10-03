"""Wave-695 homotopy-28 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w695 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_abelian": b.bench_homotopy_abelian_family(),
        "homotopy_finite": b.bench_homotopy_finite_family(),
        "homotopy_infinite": b.bench_homotopy_infinite_family(),
        "homotopy_extended": b.bench_homotopy_extended_family(),
        "stable_synthetic": b.bench_stable_synthetic_family(),
        "stable_compact": b.bench_stable_compact_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
