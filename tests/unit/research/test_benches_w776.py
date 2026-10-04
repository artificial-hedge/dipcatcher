"""Wave-776 point-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w776 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cox_process": b.bench_cox_process_family(),
        "hawkes_point": b.bench_hawkes_point_family(),
        "self_excite": b.bench_self_excite_family(),
        "marked_point": b.bench_marked_point_family(),
        "campbell_thm": b.bench_campbell_thm_family(),
        "palm_dist": b.bench_palm_dist_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
