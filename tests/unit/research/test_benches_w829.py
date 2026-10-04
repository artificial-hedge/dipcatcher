"""Wave-829 continuous-martingale adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w829 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "local_time_process": b.bench_local_time_process_family(),
        "bounded_mart": b.bench_bounded_mart_family(),
        "fv_mart": b.bench_fv_mart_family(),
        "cadlag_mart": b.bench_cadlag_mart_family(),
        "locator_proc": b.bench_locator_proc_family(),
        "decomp_mart": b.bench_decomp_mart_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
