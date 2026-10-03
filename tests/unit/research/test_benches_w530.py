"""Wave-530 nonuniform-hyperbolicity adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w530 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pesin_theory": b.bench_pesin_theory_family(),
        "nonuniform_hyp": b.bench_nonuniform_hyp_family(),
        "dominated_split": b.bench_dominated_split_family(),
        "osceledets_reg": b.bench_osceledets_reg_family(),
        "lyapunov_chart": b.bench_lyapunov_chart_family(),
        "katok_horseshoe": b.bench_katok_horseshoe_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
