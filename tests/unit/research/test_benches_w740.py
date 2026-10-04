"""Wave-740 percolation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w740 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "smirnov_percolation": b.bench_smirnov_percolation_family(),
        "duminil_copin": b.bench_duminil_copin_family(),
        "kesten_percolation": b.bench_kesten_percolation_family(),
        "cardy_formula": b.bench_cardy_formula_family(),
        "russo_seymour": b.bench_russo_seymour_family(),
        "grimmett_percolation": b.bench_grimmett_percolation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
