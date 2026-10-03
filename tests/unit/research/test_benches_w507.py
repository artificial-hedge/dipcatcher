"""Wave-507 quasi-category/Joyal adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w507 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quasi_cat": b.bench_quasi_cat_family(),
        "joyal_model": b.bench_joyal_model_family(),
        "homotopy_coherent": b.bench_homotopy_coherent_family(),
        "nerve_quasi": b.bench_nerve_quasi_family(),
        "htc_colimit": b.bench_htc_colimit_family(),
        "marking_qcat": b.bench_marking_qcat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
