"""Wave-738 Brownian-map adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w738 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "marckert_mokkadem": b.bench_marckert_mokkadem_family(),
        "le_gall_miermont": b.bench_le_gall_miermont_family(),
        "curien_legall": b.bench_curien_legall_family(),
        "abraham_bipartite": b.bench_abraham_bipartite_family(),
        "bettinelli_jacob": b.bench_bettinelli_jacob_family(),
        "chapuy_dolega": b.bench_chapuy_dolega_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
