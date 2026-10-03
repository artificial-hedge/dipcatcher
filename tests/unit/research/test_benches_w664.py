"""Wave-664 spectral-AG-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w664 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_moduli": b.bench_spectral_moduli_family(),
        "e_ring_moduli": b.bench_e_ring_moduli_family(),
        "tmf_stack": b.bench_tmf_stack_family(),
        "spectral_artstack": b.bench_spectral_artstack_family(),
        "structured_spec": b.bench_structured_spec_family(),
        "elliptic_spec2": b.bench_elliptic_spec2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
