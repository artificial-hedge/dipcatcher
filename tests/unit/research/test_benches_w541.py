"""Wave-541 complex-analysis-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w541 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "riemann_mapping": b.bench_riemann_mapping_family(),
        "schwarz_lemma": b.bench_schwarz_lemma_family(),
        "picard_thm": b.bench_picard_thm_family(),
        "montel_normal": b.bench_montel_normal_family(),
        "runge_approx": b.bench_runge_approx_family(),
        "jensen_formula": b.bench_jensen_formula_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
