"""Wave-786 rough-path adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w786 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "rough_path": b.bench_rough_path_family(),
        "signature_transform2": b.bench_signature_transform2_family(),
        "controlled_path": b.bench_controlled_path_family(),
        "lyons_lift": b.bench_lyons_lift_family(),
        "hairspring_map": b.bench_hairspring_map_family(),
        "area_mart": b.bench_area_mart_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
