"""Wave-553 mirror-symmetry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w553 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mirror_symmetry": b.bench_mirror_symmetry_family(),
        "givental_j": b.bench_givental_j_family(),
        "quantum_cohomology": b.bench_quantum_cohomology_family(),
        "quintic_invariants": b.bench_quintic_invariants_family(),
        "toric_mirror": b.bench_toric_mirror_family(),
        "frobenius_mfd": b.bench_frobenius_mfd_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
