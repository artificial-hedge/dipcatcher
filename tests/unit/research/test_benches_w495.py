"""Wave-495 infinity-2-category adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w495 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "globular_model": b.bench_globular_model_family(),
        "opetopic": b.bench_opetopic_family(),
        "theta_space": b.bench_theta_space_family(),
        "complicial": b.bench_complicial_family(),
        "verity_gray": b.bench_verity_gray_family(),
        "weak_infty": b.bench_weak_infty_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
