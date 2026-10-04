"""Wave-556 Teichmueller adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w556 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "weil_petersson": b.bench_weil_petersson_family(),
        "mapping_class": b.bench_mapping_class_family(),
        "quadratic_diff": b.bench_quadratic_diff_family(),
        "earthquake_map": b.bench_earthquake_map_family(),
        "extremal_length": b.bench_extremal_length_family(),
        "pseudo_anosov": b.bench_pseudo_anosov_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
