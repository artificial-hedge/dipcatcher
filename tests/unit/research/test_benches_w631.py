"""Wave-631 etale-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w631 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_cover3": b.bench_etale_cover3_family(),
        "etale_site3": b.bench_etale_site3_family(),
        "constructible_sh": b.bench_constructible_sh_family(),
        "weil_sheaf": b.bench_weil_sheaf_family(),
        "torsion_sheaf": b.bench_torsion_sheaf_family(),
        "ql_sheaf": b.bench_ql_sheaf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
