"""Wave-536 symplectic-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w536 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "symplectic_form": b.bench_symplectic_form_family(),
        "lagrangian_mfd": b.bench_lagrangian_mfd_family(),
        "hamiltonian_flow": b.bench_hamiltonian_flow_family(),
        "poisson_bracket": b.bench_poisson_bracket_family(),
        "contact_geom": b.bench_contact_geom_family(),
        "gromov_nonsq": b.bench_gromov_nonsq_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
