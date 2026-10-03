"""Wave-460 condensed-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w460 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "analytic_ring2": b.bench_analytic_ring2_family(),
        "solid_tensor": b.bench_solid_tensor_family(),
        "trace_class": b.bench_trace_class_family(),
        "clausen_scholze": b.bench_clausen_scholze_family(),
        "solid_derived": b.bench_solid_derived_family(),
        "pyknotic": b.bench_pyknotic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
