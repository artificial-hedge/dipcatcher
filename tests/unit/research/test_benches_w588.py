"""Wave-588 homotopy-11 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w588 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ehp_sequence": b.bench_ehp_sequence_family(),
        "james_period": b.bench_james_period_family(),
        "whitehead_prod": b.bench_whitehead_prod_family(),
        "freudenthal_susp": b.bench_freudenthal_susp_family(),
        "moore_space": b.bench_moore_space_family(),
        "unstable_adams": b.bench_unstable_adams_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
