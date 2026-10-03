"""Wave-703 homotopy-30 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w703 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_suspension2": b.bench_homotopy_suspension2_family(),
        "homotopy_fiber3": b.bench_homotopy_fiber3_family(),
        "stable_derivator": b.bench_stable_derivator_family(),
        "homotopy_spectrum2": b.bench_homotopy_spectrum2_family(),
        "stable_excisive": b.bench_stable_excisive_family(),
        "homotopy_vn": b.bench_homotopy_vn_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
