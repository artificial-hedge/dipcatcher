"""Wave-660 chromatic-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w660 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "morava_k2": b.bench_morava_k2_family(),
        "telescope_tower2": b.bench_telescope_tower2_family(),
        "chromatic_l2": b.bench_chromatic_l2_family(),
        "picard_spec": b.bench_picard_spec_family(),
        "periodicity_height": b.bench_periodicity_height_family(),
        "chromatic_completion": b.bench_chromatic_completion_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
