"""Wave-666 derived-geometry-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w666 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_abelian": b.bench_derived_abelian_family(),
        "simplicial_comm": b.bench_simplicial_comm_family(),
        "derived_bezout": b.bench_derived_bezout_family(),
        "derived_hecke": b.bench_derived_hecke_family(),
        "cotangent_stack": b.bench_cotangent_stack_family(),
        "derived_bun": b.bench_derived_bun_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
