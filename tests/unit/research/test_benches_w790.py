"""Wave-790 McKean-Vlasov adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w790 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mckean_vlasov": b.bench_mckean_vlasov_family(),
        "mean_field_game2": b.bench_mean_field_game2_family(),
        "propagation_chaos": b.bench_propagation_chaos_family(),
        "kac_theorem": b.bench_kac_theorem_family(),
        "nonlinear_markov": b.bench_nonlinear_markov_family(),
        "self_stabilizing": b.bench_self_stabilizing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
