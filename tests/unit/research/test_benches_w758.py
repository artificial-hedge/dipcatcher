"""Wave-758 GFF-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w758 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "powell_gff": b.bench_powell_gff_family(),
        "aru_gff": b.bench_aru_gff_family(),
        "ding_zeitouni": b.bench_ding_zeitouni_family(),
        "chatterjee_gff": b.bench_chatterjee_gff_family(),
        "bolthausen_gff": b.bench_bolthausen_gff_family(),
        "najafi_gff": b.bench_najafi_gff_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
