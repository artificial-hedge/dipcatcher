"""Wave-768 large-deviation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w768 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "varadhan_ldp": b.bench_varadhan_ldp_family(),
        "freidlin_wentzell": b.bench_freidlin_wentzell_family(),
        "dw_ldp": b.bench_dw_ldp_family(),
        "sanov_thm": b.bench_sanov_thm_family(),
        "mogulskii_thm": b.bench_mogulskii_thm_family(),
        "schider_thm": b.bench_schider_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
