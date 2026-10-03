"""Wave-549 index-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w549 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "atiyah_singer": b.bench_atiyah_singer_family(),
        "dirac_op": b.bench_dirac_op_family(),
        "eta_invariant": b.bench_eta_invariant_family(),
        "heat_kernel2": b.bench_heat_kernel2_family(),
        "signature_op": b.bench_signature_op_family(),
        "analytic_torsion": b.bench_analytic_torsion_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
