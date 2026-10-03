"""Wave-375 algebraic-geometry-5 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w375 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "riemann_roch": b.bench_riemann_roch_family(),
        "sheaf_cohomology": b.bench_sheaf_cohomology_family(),
        "scheme_local": b.bench_scheme_local_family(),
        "blowup": b.bench_blowup_family(),
        "elliptic_group": b.bench_elliptic_group_family(),
        "moduli_stable": b.bench_moduli_stable_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
