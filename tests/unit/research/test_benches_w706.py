"""Wave-706 chromatic-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w706 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chromatic_layer": b.bench_chromatic_layer_family(),
        "morava_maven": b.bench_morava_maven_family(),
        "chromatic_square2": b.bench_chromatic_square2_family(),
        "lubin_tate3": b.bench_lubin_tate3_family(),
        "elliptic_morava": b.bench_elliptic_morava_family(),
        "chromatic_base": b.bench_chromatic_base_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
