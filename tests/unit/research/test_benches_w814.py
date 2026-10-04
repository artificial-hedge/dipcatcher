"""Wave-814 stochastic-flow adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w814 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stochastic_flow": b.bench_stochastic_flow_family(),
        "kunita_flow": b.bench_kunita_flow_family(),
        "liouville_flow": b.bench_liouville_flow_family(),
        "stochastic_damping": b.bench_stochastic_damping_family(),
        "meyers_process": b.bench_meyers_process_family(),
        "karal_flow": b.bench_karal_flow_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
