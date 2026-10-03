"""Wave-481 motivic-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w481 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mtc_motive": b.bench_mtc_motive_family(),
        "fqmotive": b.bench_fqmotive_family(),
        "triang_motive": b.bench_triang_motive_family(),
        "motivic_chern": b.bench_motivic_chern_family(),
        "motivic_landin": b.bench_motivic_landin_family(),
        "higher_chow2": b.bench_higher_chow2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
