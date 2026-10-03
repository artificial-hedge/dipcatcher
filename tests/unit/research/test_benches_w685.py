"""Wave-685 homotopy-26 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w685 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_limit": b.bench_homotopy_limit_family(),
        "homotopy_tower": b.bench_homotopy_tower_family(),
        "spectral_sequence5": b.bench_spectral_sequence5_family(),
        "homotopy_class2": b.bench_homotopy_class2_family(),
        "stable_mapping": b.bench_stable_mapping_family(),
        "stable_bousfield": b.bench_stable_bousfield_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
