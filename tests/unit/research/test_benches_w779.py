"""Wave-779 matrix-analytic adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w779 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "neuts_map": b.bench_neuts_map_family(),
        "phase_type": b.bench_phase_type_family(),
        "matrix_geom": b.bench_matrix_geom_family(),
        "quasi_birth": b.bench_quasi_birth_family(),
        "ramaswami": b.bench_ramaswami_family(),
        "logarithmic_red": b.bench_logarithmic_red_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
