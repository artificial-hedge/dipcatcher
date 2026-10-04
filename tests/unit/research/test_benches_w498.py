"""Wave-498 moduli/GW adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w498 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kuranishi": b.bench_kuranishi_family(),
        "hilbert_scheme2": b.bench_hilbert_scheme2_family(),
        "quot_scheme": b.bench_quot_scheme_family(),
        "m_bar_gn": b.bench_m_bar_gn_family(),
        "stable_map": b.bench_stable_map_family(),
        "gromov_witten": b.bench_gromov_witten_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
