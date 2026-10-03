"""Wave-802 neural-SDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w802 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "latent_sde": b.bench_latent_sde_family(),
        "neural_cde": b.bench_neural_cde_family(),
        "neural_rde": b.bench_neural_rde_family(),
        "sde_gan": b.bench_sde_gan_family(),
        "sde_matching": b.bench_sde_matching_family(),
        "logsig_rde": b.bench_logsig_rde_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
