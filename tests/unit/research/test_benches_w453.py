"""Wave-453 higher-algebra-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w453 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "operad_koszul": b.bench_operad_koszul_family(),
        "bar_cobar": b.bench_bar_cobar_family(),
        "factor_homology": b.bench_factor_homology_family(),
        "hochschild_hom": b.bench_hochschild_hom_family(),
        "deligne_conj": b.bench_deligne_conj_family(),
        "primitive_elts": b.bench_primitive_elts_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
