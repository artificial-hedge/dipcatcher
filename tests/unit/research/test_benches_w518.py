"""Wave-518 Yang-Baxter/braid adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w518 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "yang_baxter": b.bench_yang_baxter_family(),
        "braid_rep": b.bench_braid_rep_family(),
        "yangian": b.bench_yangian_family(),
        "rtt_formalism": b.bench_rtt_formalism_family(),
        "quantum_double": b.bench_quantum_double_family(),
        "ribbon_cat": b.bench_ribbon_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
