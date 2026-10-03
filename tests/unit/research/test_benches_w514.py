"""Wave-514 differential-cohomology adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w514 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "diff_cohom": b.bench_diff_cohom_family(),
        "cheeger_simons": b.bench_cheeger_simons_family(),
        "deligne_cohom": b.bench_deligne_cohom_family(),
        "flat_bundle": b.bench_flat_bundle_family(),
        "beilinson_reg": b.bench_beilinson_reg_family(),
        "secondary_inv": b.bench_secondary_inv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
