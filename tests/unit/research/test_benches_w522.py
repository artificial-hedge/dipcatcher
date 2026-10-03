"""Wave-522 additive-combinatorics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w522 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "freiman_thm": b.bench_freiman_thm_family(),
        "szemeredi": b.bench_szemeredi_family(),
        "green_tao": b.bench_green_tao_family(),
        "roth_thm": b.bench_roth_thm_family(),
        "gowers_norm": b.bench_gowers_norm_family(),
        "plunnecke": b.bench_plunnecke_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
