"""Wave-772 branching-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w772 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "galton_watson": b.bench_galton_watson_family(),
        "branching_imm": b.bench_branching_imm_family(),
        "multi_type_branch": b.bench_multi_type_branch_family(),
        "crump_mode": b.bench_crump_mode_family(),
        "kimmel_branch": b.bench_kimmel_branch_family(),
        "sevastyanov": b.bench_sevastyanov_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
