"""Wave-542 several-complex-variables adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w542 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hartogs_thm": b.bench_hartogs_thm_family(),
        "domain_holo": b.bench_domain_holo_family(),
        "pseudoconvex": b.bench_pseudoconvex_family(),
        "levi_problem": b.bench_levi_problem_family(),
        "oka_coherence": b.bench_oka_coherence_family(),
        "d_bar_neumann": b.bench_d_bar_neumann_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
