"""Wave-503 elliptic-surface adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w503 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "elliptic_surface": b.bench_elliptic_surface_family(),
        "weierstrass_eq": b.bench_weierstrass_eq_family(),
        "kodaira_fiber": b.bench_kodaira_fiber_family(),
        "tate_algorithm": b.bench_tate_algorithm_family(),
        "mordell_weil2": b.bench_mordell_weil2_family(),
        "neron_model": b.bench_neron_model_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
