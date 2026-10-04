"""Wave-419 algebraic-topology-5 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w419 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "serre_fibration": b.bench_serre_fibration_family(),
        "path_fibration": b.bench_path_fibration_family(),
        "bundle_section": b.bench_bundle_section_family(),
        "classify_space": b.bench_classify_space_family(),
        "vector_bundle": b.bench_vector_bundle_family(),
        "thom_space": b.bench_thom_space_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
