"""Wave-496 DAG-deformation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w496 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_deformation": b.bench_derived_deformation_family(),
        "formal_deformation": b.bench_formal_deformation_family(),
        "dag_representation": b.bench_dag_representation_family(),
        "derived_moduli": b.bench_derived_moduli_family(),
        "tangent_coh": b.bench_tangent_coh_family(),
        "obstruction_2": b.bench_obstruction_2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
