"""Wave-334 secure-computation adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w334 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "garbled_circuit": b.bench_garbled_circuit_family(),
        "bgw_mpc": b.bench_bgw_mpc_family(),
        "beaver_triple": b.bench_beaver_triple_family(),
        "ot_extension": b.bench_ot_extension_family(),
        "spdz_mac": b.bench_spdz_mac_family(),
        "psi_intersect": b.bench_psi_intersect_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
