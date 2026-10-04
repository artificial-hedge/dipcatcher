"""Wave-527 hyperbolic-dynamics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w527 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "anosov": b.bench_anosov_family(),
        "srb_measure": b.bench_srb_measure_family(),
        "horseshoe": b.bench_horseshoe_family(),
        "stable_mfld": b.bench_stable_mfld_family(),
        "bowen_spec": b.bench_bowen_spec_family(),
        "markov_partition": b.bench_markov_partition_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
