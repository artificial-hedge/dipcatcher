"""Wave-472 motivic-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w472 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "alg_cobordism": b.bench_alg_cobordism_family(),
        "hermitian_k": b.bench_hermitian_k_family(),
        "oriented_coh": b.bench_oriented_coh_family(),
        "slice_spec": b.bench_slice_spec_family(),
        "motivic_stem2": b.bench_motivic_stem2_family(),
        "rostmotive": b.bench_rostmotive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
