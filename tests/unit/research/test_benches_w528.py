"""Wave-528 thermodynamic-formalism adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w528 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "transfer_op": b.bench_transfer_op_family(),
        "thermo_formal": b.bench_thermo_formal_family(),
        "pressure_thm": b.bench_pressure_thm_family(),
        "equilibrium_state": b.bench_equilibrium_state_family(),
        "ruelle_zeta": b.bench_ruelle_zeta_family(),
        "lasota_yorke": b.bench_lasota_yorke_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
