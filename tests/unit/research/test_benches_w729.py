"""Wave-729 motivic-A1-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w729 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "roald_suslin": b.bench_roald_suslin_family(),
        "jogiad_motive": b.bench_jogiad_motive_family(),
        "hauwas_nori": b.bench_hauwas_nori_family(),
        "motivic_pipe": b.bench_motivic_pipe_family(),
        "thom_mgl2": b.bench_thom_mgl2_family(),
        "voev_suslin": b.bench_voev_suslin_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
