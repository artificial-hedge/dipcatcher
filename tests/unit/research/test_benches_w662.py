"""Wave-662 higher-algebra-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w662 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bar_resolution2": b.bench_bar_resolution2_family(),
        "hochschild_hom2": b.bench_hochschild_hom2_family(),
        "factor_homology2": b.bench_factor_homology2_family(),
        "deligne_conj2": b.bench_deligne_conj2_family(),
        "braces_higher": b.bench_braces_higher_family(),
        "little_cubes": b.bench_little_cubes_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
