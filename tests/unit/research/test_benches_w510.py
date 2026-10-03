"""Wave-510 spectral-AG adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w510 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_scheme2": b.bench_spectral_scheme2_family(),
        "connective_e_ring": b.bench_connective_e_ring_family(),
        "spectral_alg": b.bench_spectral_alg_family(),
        "spectral_stack": b.bench_spectral_stack_family(),
        "elliptic_cohor": b.bench_elliptic_cohor_family(),
        "taf_lurie": b.bench_taf_lurie_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
