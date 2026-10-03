"""Wave-319 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w319 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "congruence_closure": b.bench_congruence_closure_family(),
        "ring_normalize": b.bench_ring_normalize_family(),
        "omega_lia": b.bench_omega_lia_family(),
        "nelson_oppen": b.bench_nelson_oppen_family(),
        "term_rewrite": b.bench_term_rewrite_family(),
        "tseitin_cnf": b.bench_tseitin_cnf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
