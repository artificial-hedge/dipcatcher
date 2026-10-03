"""Wave-486 algebraic-K-theory-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w486 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "waldhausen_k": b.bench_waldhausen_k_family(),
        "plus_k": b.bench_plus_k_family(),
        "kv_theory": b.bench_kv_theory_family(),
        "karoubi_v": b.bench_karoubi_v_family(),
        "vorst_stab": b.bench_vorst_stab_family(),
        "nk_theory": b.bench_nk_theory_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
