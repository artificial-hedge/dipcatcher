"""Wave-585 intersection-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w585 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "intersection_theory": b.bench_intersection_theory_family(),
        "macpherson_chern": b.bench_macpherson_chern_family(),
        "weil_divisor": b.bench_weil_divisor_family(),
        "picard_group": b.bench_picard_group_family(),
        "line_bundle": b.bench_line_bundle_family(),
        "canonical_bundle": b.bench_canonical_bundle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
