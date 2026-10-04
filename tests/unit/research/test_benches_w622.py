"""Wave-622 prismatic adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w622 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "prism2": b.bench_prism2_family(),
        "prismatic_site": b.bench_prismatic_site_family(),
        "delta_ring": b.bench_delta_ring_family(),
        "prismatic_crystal": b.bench_prismatic_crystal_family(),
        "hodge_tate": b.bench_hodge_tate_family(),
        "nygaard2": b.bench_nygaard2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
