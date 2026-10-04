"""Wave-439 formal-groups/chromatic adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w439 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "formal_group": b.bench_formal_group_family(),
        "lazard_ring": b.bench_lazard_ring_family(),
        "formal_module": b.bench_formal_module_family(),
        "height_strata": b.bench_height_strata_family(),
        "lubin_tate": b.bench_lubin_tate_family(),
        "morava_k": b.bench_morava_k_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
