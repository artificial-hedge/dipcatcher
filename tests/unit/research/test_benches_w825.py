"""Wave-825 measurable-selection adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w825 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "measur_select": b.bench_measur_select_family(),
        "kura_ryll": b.bench_kura_ryll_family(),
        "castaing_rep": b.bench_castaing_rep_family(),
        "measurable_graph": b.bench_measurable_graph_family(),
        "integrand_map": b.bench_integrand_map_family(),
        "stoch_open": b.bench_stoch_open_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
