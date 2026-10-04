"""Wave-570 positivity/moduli adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w570 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hodge_index": b.bench_hodge_index_family(),
        "kodaira_vanishing": b.bench_kodaira_vanishing_family(),
        "kollar_mori": b.bench_kollar_mori_family(),
        "boundedness_moduli": b.bench_boundedness_moduli_family(),
        "stability_sheaf": b.bench_stability_sheaf_family(),
        "bogomolov_ineq": b.bench_bogomolov_ineq_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
