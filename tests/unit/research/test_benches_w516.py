"""Wave-516 Kac-Moody/VOA adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w516 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kac_moody": b.bench_kac_moody_family(),
        "weyl_kac": b.bench_weyl_kac_family(),
        "vertex_alg": b.bench_vertex_alg_family(),
        "moonshine_module": b.bench_moonshine_module_family(),
        "affine_lie": b.bench_affine_lie_family(),
        "zhu_algebra": b.bench_zhu_algebra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
