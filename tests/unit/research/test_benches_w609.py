"""Wave-609 spectral-AG-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w609 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e_infty_space": b.bench_e_infty_space_family(),
        "brave_new_ring": b.bench_brave_new_ring_family(),
        "thom_constr": b.bench_thom_constr_family(),
        "log_ring": b.bench_log_ring_family(),
        "orient_cohom": b.bench_orient_cohom_family(),
        "formal_moduli": b.bench_formal_moduli_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
