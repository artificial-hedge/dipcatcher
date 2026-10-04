"""Wave-497 birational-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w497 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "minimal_model": b.bench_minimal_model_family(),
        "klt_pair": b.bench_klt_pair_family(),
        "flip_cone": b.bench_flip_cone_family(),
        "fano_mori": b.bench_fano_mori_family(),
        "mmp_algorithm": b.bench_mmp_algorithm_family(),
        "toric_flip": b.bench_toric_flip_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
