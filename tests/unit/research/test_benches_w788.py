"""Wave-788 Malliavin-calculus adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w788 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "clark_ocone": b.bench_clark_ocone_family(),
        "nualart_pardoux": b.bench_nualart_pardoux_family(),
        "divergence_op": b.bench_divergence_op_family(),
        "wiener_chaos": b.bench_wiener_chaos_family(),
        "skorohod_int": b.bench_skorohod_int_family(),
        "nourdin_peccati": b.bench_nourdin_peccati_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
