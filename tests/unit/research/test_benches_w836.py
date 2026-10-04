"""Wave-836 integral-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w836 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "crofton_formula": b.bench_crofton_formula_family(),
        "kinematic_measure": b.bench_kinematic_measure_family(),
        "buffon_needle": b.bench_buffon_needle_family(),
        "santalo_measure": b.bench_santalo_measure_family(),
        "kubota_mean_width": b.bench_kubota_mean_width_family(),
        "hadwiger_chars": b.bench_hadwiger_chars_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
