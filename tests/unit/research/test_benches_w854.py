"""Wave-854 radial-basis-function adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w854 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "rbf_interp": b.bench_rbf_interp_family(),
        "gaussian_rbf": b.bench_gaussian_rbf_family(),
        "multiquadric_rbf": b.bench_multiquadric_rbf_family(),
        "kansa_collocation": b.bench_kansa_collocation_family(),
        "rbf_finite_diff": b.bench_rbf_finite_diff_family(),
        "wendland_rbf": b.bench_wendland_rbf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
