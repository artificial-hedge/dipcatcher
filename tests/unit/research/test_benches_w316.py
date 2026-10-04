"""Wave-316 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w316 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "trotter_suzuki": b.bench_trotter_suzuki_family(),
        "qdrift": b.bench_qdrift_family(),
        "shadow_tomography": b.bench_shadow_tomography_family(),
        "vqd_states": b.bench_vqd_states_family(),
        "adapt_vqe": b.bench_adapt_vqe_family(),
        "hhl_lite": b.bench_hhl_lite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
