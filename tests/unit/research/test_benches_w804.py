"""Wave-804 Malliavin adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w804 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "nualart_zakai": b.bench_nualart_zakai_family(),
        "watanabe_map": b.bench_watanabe_map_family(),
        "malliavin_cov": b.bench_malliavin_cov_family(),
        "density_bound": b.bench_density_bound_family(),
        "absolute_cont": b.bench_absolute_cont_family(),
        "smoothness_h": b.bench_smoothness_h_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
