"""Wave-466 sheaf-3/microlocal adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w466 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "micro_supp": b.bench_micro_supp_family(),
        "kashiwara_schapira": b.bench_kashiwara_schapira_family(),
        "loc_system": b.bench_loc_system_family(),
        "perverse_2": b.bench_perverse_2_family(),
        "stacky_sheaf": b.bench_stacky_sheaf_family(),
        "sheaf_homotopy": b.bench_sheaf_homotopy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
