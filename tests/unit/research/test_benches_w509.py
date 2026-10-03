"""Wave-509 Weil-II/l-adic adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w509 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_site2": b.bench_etale_site2_family(),
        "l_adic_sheaf": b.bench_l_adic_sheaf_family(),
        "frobenius_action": b.bench_frobenius_action_family(),
        "groth_lefschetz": b.bench_groth_lefschetz_family(),
        "deligne_weil2": b.bench_deligne_weil2_family(),
        "purity_thm": b.bench_purity_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
