"""Wave-670 chromatic-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w670 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "morava_k3": b.bench_morava_k3_family(),
        "morava_e2": b.bench_morava_e2_family(),
        "chromatic_l3": b.bench_chromatic_l3_family(),
        "telescope_tower3": b.bench_telescope_tower3_family(),
        "picard_spec2": b.bench_picard_spec2_family(),
        "red_shift2": b.bench_red_shift2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
