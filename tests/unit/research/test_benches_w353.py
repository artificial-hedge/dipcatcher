"""Wave-353 ODE-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w353 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "picard_lindelof": b.bench_picard_lindelof_family(),
        "gronwall_lemma": b.bench_gronwall_lemma_family(),
        "sturm_liouville": b.bench_sturm_liouville_family(),
        "phase_plane": b.bench_phase_plane_family(),
        "lyapunov_stability": b.bench_lyapunov_stability_family(),
        "variation_params": b.bench_variation_params_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
