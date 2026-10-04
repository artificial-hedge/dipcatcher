"""Wave-524 Ramsey-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w524 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hales_jewett": b.bench_hales_jewett_family(),
        "rado_thm": b.bench_rado_thm_family(),
        "gallai_thm": b.bench_gallai_thm_family(),
        "schur_thm": b.bench_schur_thm_family(),
        "hindman": b.bench_hindman_family(),
        "furstenberg": b.bench_furstenberg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
