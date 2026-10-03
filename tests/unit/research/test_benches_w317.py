"""Wave-317 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w317 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "canny_edge": b.bench_canny_edge_family(),
        "otsu_threshold": b.bench_otsu_threshold_family(),
        "watershed_seg": b.bench_watershed_seg_family(),
        "slic_superpixels": b.bench_slic_superpixels_family(),
        "nlm_denoise": b.bench_nlm_denoise_family(),
        "distance_transform": b.bench_distance_transform_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
