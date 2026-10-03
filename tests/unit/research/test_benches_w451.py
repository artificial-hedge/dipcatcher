"""Wave-451 equivariant-homotopy adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w451 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "g_spectrum": b.bench_g_spectrum_family(),
        "mackey_functor": b.bench_mackey_functor_family(),
        "norm_map": b.bench_norm_map_family(),
        "ro_grading": b.bench_ro_grading_family(),
        "wirthmuller": b.bench_wirthmuller_family(),
        "tom_dieck": b.bench_tom_dieck_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
