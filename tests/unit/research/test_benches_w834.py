"""Wave-834 Markov-semigroup adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w834 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dirichlet_form": b.bench_dirichlet_form_family(),
        "markov_semigroup": b.bench_markov_semigroup_family(),
        "poincare_semigroup": b.bench_poincare_semigroup_family(),
        "log_sobolev_sem": b.bench_log_sobolev_sem_family(),
        "hypercontractive": b.bench_hypercontractive_family(),
        "spectral_gap_sem": b.bench_spectral_gap_sem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
