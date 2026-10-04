"""Wave-736 LQG adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w736 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sheffield_gff": b.bench_sheffield_gff_family(),
        "berestycki_sheffield": b.bench_berestycki_sheffield_family(),
        "aru_powell": b.bench_aru_powell_family(),
        "huang_rhodes": b.bench_huang_rhodes_family(),
        "bisbisot_sheffield": b.bench_bisbisot_sheffield_family(),
        "dhms_lqg": b.bench_dhms_lqg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
