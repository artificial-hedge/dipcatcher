"""Wave-569 geometric-PDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w569 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "yamabe_problem": b.bench_yamabe_problem_family(),
        "prescribed_curvature": b.bench_prescribed_curvature_family(),
        "nirenberg_problem": b.bench_nirenberg_problem_family(),
        "kazdan_warner": b.bench_kazdan_warner_family(),
        "aubin_thm": b.bench_aubin_thm_family(),
        "trudinger_thm": b.bench_trudinger_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
