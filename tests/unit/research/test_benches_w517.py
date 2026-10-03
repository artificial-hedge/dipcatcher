"""Wave-517 quantum-groups adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w517 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quantum_group": b.bench_quantum_group_family(),
        "crystal_base": b.bench_crystal_base_family(),
        "quantum_rmatrix": b.bench_quantum_rmatrix_family(),
        "jimbo_drin": b.bench_jimbo_drin_family(),
        "lusztig_can": b.bench_lusztig_can_family(),
        "quantum_schur": b.bench_quantum_schur_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
