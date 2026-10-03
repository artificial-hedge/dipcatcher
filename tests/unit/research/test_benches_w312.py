"""Wave-312 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w312 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hlc_clock": b.bench_hlc_clock_family(),
        "delta_crdt": b.bench_delta_crdt_family(),
        "raft_log": b.bench_raft_log_family(),
        "bracha_bcast": b.bench_bracha_bcast_family(),
        "tot_order": b.bench_tot_order_family(),
        "quorum_weighted": b.bench_quorum_weighted_family(),
        "abd_register": b.bench_abd_register_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
