"""Wave-746 ASEP adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w746 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "liggett_exclusion": b.bench_liggett_exclusion_family(),
        "spitzer_exclusion": b.bench_spitzer_exclusion_family(),
        "sasamoto_tasep": b.bench_sasamoto_tasep_family(),
        "tracy_widom_tasep": b.bench_tracy_widom_tasep_family(),
        "derrida_tasep": b.bench_derrida_tasep_family(),
        "ferrari_tasep": b.bench_ferrari_tasep_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
