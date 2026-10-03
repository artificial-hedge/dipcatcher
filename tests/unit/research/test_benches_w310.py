"""Wave-310 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w310 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gottesman_knill": b.bench_gottesman_knill_family(),
        "steane_code": b.bench_steane_code_family(),
        "surface_code": b.bench_surface_code_family(),
        "shor_code": b.bench_shor_code_family(),
        "syndrome_circuit": b.bench_syndrome_circuit_family(),
        "repetition_qec": b.bench_repetition_qec_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
