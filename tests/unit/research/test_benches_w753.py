"""Wave-753 O(N)-model adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w753 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fernandez_frohlich": b.bench_fernandez_frohlich_family(),
        "aizenman_irf": b.bench_aizenman_irf_family(),
        "fradkin_sokal": b.bench_fradkin_sokal_family(),
        "nienhuis_on": b.bench_nienhuis_on_family(),
        "cardy_on": b.bench_cardy_on_family(),
        "pelissetto_vicari": b.bench_pelissetto_vicari_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
