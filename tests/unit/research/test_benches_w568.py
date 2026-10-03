"""Wave-568 symplectic-field-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w568 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "symplectic_field": b.bench_symplectic_field_family(),
        "contact_homology3": b.bench_contact_homology3_family(),
        "floer_homol": b.bench_floer_homol_family(),
        "reeb_orbit": b.bench_reeb_orbit_family(),
        "sft_algebra": b.bench_sft_algebra_family(),
        "eliashberg_givental": b.bench_eliashberg_givental_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
