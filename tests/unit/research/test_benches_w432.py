"""Wave-432 spectral-sequences-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w432 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "atiyah_hirzebruch": b.bench_atiyah_hirzebruch_family(),
        "serre_ss3": b.bench_serre_ss3_family(),
        "leary_ss": b.bench_leary_ss_family(),
        "descent_ss": b.bench_descent_ss_family(),
        "motivic_ss": b.bench_motivic_ss_family(),
        "vanishing_ss": b.bench_vanishing_ss_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
