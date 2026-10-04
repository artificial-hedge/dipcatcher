"""Wave-807 optimal-stopping adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w807 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "snell_envelope": b.bench_snell_envelope_family(),
        "secretary_dp": b.bench_secretary_dp_family(),
        "cayley_moser": b.bench_cayley_moser_family(),
        "chow_robbins": b.bench_chow_robbins_family(),
        "markov_stopping": b.bench_markov_stopping_family(),
        "free_boundary": b.bench_free_boundary_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
