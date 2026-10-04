"""Wave-561 minimal-surfaces adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w561 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "minimal_surface": b.bench_minimal_surface_family(),
        "plateau_problem": b.bench_plateau_problem_family(),
        "brakke_flow": b.bench_brakke_flow_family(),
        "almgren_pitts": b.bench_almgren_pitts_family(),
        "simon_regularity": b.bench_simon_regularity_family(),
        "stable_minimal": b.bench_stable_minimal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
