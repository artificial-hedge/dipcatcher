"""Wave-525 ergodic-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w525 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "birkhoff": b.bench_birkhoff_family(),
        "mean_ergodic": b.bench_mean_ergodic_family(),
        "mixing_weak": b.bench_mixing_weak_family(),
        "entropy_ks": b.bench_entropy_ks_family(),
        "bernoulli_shift": b.bench_bernoulli_shift_family(),
        "osceledets": b.bench_osceledets_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
