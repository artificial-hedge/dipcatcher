"""Wave-859 meshfree/moving-least-squares adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w859 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "moving_least_sq": b.bench_moving_least_sq_family(),
        "mls_shape": b.bench_mls_shape_family(),
        "hp_clouds": b.bench_hp_clouds_family(),
        "meshless_local": b.bench_meshless_local_family(),
        "point_cloud_interp": b.bench_point_cloud_interp_family(),
        "diffuse_element": b.bench_diffuse_element_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
