"""Wave-356 stochastic-processes-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w356 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "markov_chain": b.bench_markov_chain_family(),
        "martingale_check": b.bench_martingale_check_family(),
        "poisson_process": b.bench_poisson_process_family(),
        "gambler_ruin": b.bench_gambler_ruin_family(),
        "stopping_time": b.bench_stopping_time_family(),
        "markov_hitting": b.bench_markov_hitting_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
