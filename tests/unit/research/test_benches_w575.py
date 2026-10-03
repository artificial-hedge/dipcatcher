"""Wave-575 random-matrix-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w575 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "circular_law": b.bench_circular_law_family(),
        "dyson_brownian": b.bench_dyson_brownian_family(),
        "sine_kernel": b.bench_sine_kernel_family(),
        "airy_process": b.bench_airy_process_family(),
        "tracy_widom": b.bench_tracy_widom_family(),
        "beta_ensemble": b.bench_beta_ensemble_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
