"""Wave-586 duality-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w586 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "groth_duality": b.bench_groth_duality_family(),
        "dualizing_cmplx": b.bench_dualizing_cmplx_family(),
        "residue_thm": b.bench_residue_thm_family(),
        "verdier_duality": b.bench_verdier_duality_family(),
        "dualizing_sheaf": b.bench_dualizing_sheaf_family(),
        "relative_duality": b.bench_relative_duality_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
