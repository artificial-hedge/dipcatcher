"""Wave-780 queueing-network adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w780 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bcmp_net": b.bench_bcmp_net_family(),
        "mean_value": b.bench_mean_value_family(),
        "convoy_net": b.bench_convoy_net_family(),
        "insensitive_thm": b.bench_insensitive_thm_family(),
        "kaufman_roberts": b.bench_kaufman_roberts_family(),
        "orku_loss": b.bench_orku_loss_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
