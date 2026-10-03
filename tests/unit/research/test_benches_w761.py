"""Wave-761 renewal-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w761 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "blackwell_renewal": b.bench_blackwell_renewal_family(),
        "key_renewal": b.bench_key_renewal_family(),
        "excess_renewal": b.bench_excess_renewal_family(),
        "alternating_renewal": b.bench_alternating_renewal_family(),
        "renewal_reward2": b.bench_renewal_reward2_family(),
        "delayed_renewal": b.bench_delayed_renewal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
