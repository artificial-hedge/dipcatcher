"""Wave-377 operad adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w377 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "operad_assoc": b.bench_operad_assoc_family(),
        "operad_comm": b.bench_operad_comm_family(),
        "little_discs": b.bench_little_discs_family(),
        "operad_tree": b.bench_operad_tree_family(),
        "endomorphism_op": b.bench_endomorphism_op_family(),
        "may_recognition": b.bench_may_recognition_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
