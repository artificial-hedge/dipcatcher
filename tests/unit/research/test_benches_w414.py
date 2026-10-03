"""Wave-414 homotopy-6 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w414 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "exact_couple": b.bench_exact_couple_family(),
        "adams_ss": b.bench_adams_ss_family(),
        "stable_homotopy": b.bench_stable_homotopy_family(),
        "whitehead_thm": b.bench_whitehead_thm_family(),
        "obstruction": b.bench_obstruction_family(),
        "cofiber": b.bench_cofiber_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
