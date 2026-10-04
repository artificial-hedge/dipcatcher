"""Wave-809 Markov-chain adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w809 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "doeblin_coupling": b.bench_doeblin_coupling_family(),
        "harris_recurrent": b.bench_harris_recurrent_family(),
        "ergodic_markov": b.bench_ergodic_markov_family(),
        "mixing_time": b.bench_mixing_time_family(),
        "drift_lyapunov": b.bench_drift_lyapunov_family(),
        "cutoff_phenomenon": b.bench_cutoff_phenomenon_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
