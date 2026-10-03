"""Wave-748 vertex-model adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w748 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "borodin_sixv": b.bench_borodin_sixv_family(),
        "gowers_knot": b.bench_gowers_knot_family(),
        "baxter_vertex": b.bench_baxter_vertex_family(),
        "reshetikhin_vertex": b.bench_reshetikhin_vertex_family(),
        "corwin_petrov": b.bench_corwin_petrov_family(),
        "aggarwal_sixv": b.bench_aggarwal_sixv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
