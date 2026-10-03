"""Wave-765 LIL/LLN adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w765 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "strassen_lil": b.bench_strassen_lil_family(),
        "chung_lil": b.bench_chung_lil_family(),
        "kolmogorov_3series": b.bench_kolmogorov_3series_family(),
        "khintchine_lln": b.bench_khintchine_lln_family(),
        "levy_convergence": b.bench_levy_convergence_family(),
        "glivenko_cantelli": b.bench_glivenko_cantelli_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
