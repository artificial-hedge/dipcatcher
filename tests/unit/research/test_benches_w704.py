"""Wave-704 higher-algebra-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w704 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "higher_algebra9": b.bench_higher_algebra9_family(),
        "operad_infty5": b.bench_operad_infty5_family(),
        "operad_swiss4": b.bench_operad_swiss4_family(),
        "koszul_duality3": b.bench_koszul_duality3_family(),
        "braces_e5": b.bench_braces_e5_family(),
        "delooping3": b.bench_delooping3_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
