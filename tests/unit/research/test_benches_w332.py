"""Wave-332 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w332 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "density_matrix": b.bench_density_matrix_family(),
        "povm_measure": b.bench_povm_measure_family(),
        "qchannel": b.bench_qchannel_family(),
        "entanglement": b.bench_entanglement_family(),
        "bell_ineq": b.bench_bell_ineq_family(),
        "state_tomo": b.bench_state_tomo_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
