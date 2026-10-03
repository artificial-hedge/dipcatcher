"""Wave-327 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w327 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "r1cs_check": b.bench_r1cs_check_family(),
        "qap_encode": b.bench_qap_encode_family(),
        "kzg_commit": b.bench_kzg_commit_family(),
        "bulletproof_ip": b.bench_bulletproof_ip_family(),
        "plonkish_gate": b.bench_plonkish_gate_family(),
        "snark_circuit": b.bench_snark_circuit_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
