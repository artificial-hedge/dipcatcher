"""Wave-454 intersection-cohomology-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w454 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ic_stalk": b.bench_ic_stalk_family(),
        "decomp_thm": b.bench_decomp_thm_family(),
        "riemann_hilbert": b.bench_riemann_hilbert_family(),
        "fourier_sato": b.bench_fourier_sato_family(),
        "vanishing_cycles": b.bench_vanishing_cycles_family(),
        "middle_ext": b.bench_middle_ext_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
