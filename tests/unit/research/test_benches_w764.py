"""Wave-764 empirical-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w764 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wiener_measure": b.bench_wiener_measure_family(),
        "dz_invariance": b.bench_dz_invariance_family(),
        "donsker_thm": b.bench_donsker_thm_family(),
        "empirical_process": b.bench_empirical_process_family(),
        "donsker_class": b.bench_donsker_class_family(),
        "osj_metric": b.bench_osj_metric_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
