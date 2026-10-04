"""Wave-832 weak-convergence-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w832 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "porte_manteau": b.bench_porte_manteau_family(),
        "continuous_map": b.bench_continuous_map_family(),
        "delta_method": b.bench_delta_method_family(),
        "skorohod_embed": b.bench_skorohod_embed_family(),
        "kmt_approx": b.bench_kmt_approx_family(),
        "empirical_bridge": b.bench_empirical_bridge_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
