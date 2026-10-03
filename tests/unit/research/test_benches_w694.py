"""Wave-694 motivic-21 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w694 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_trace": b.bench_motivic_trace_family(),
        "motivic_transfer2": b.bench_motivic_transfer2_family(),
        "motivic_coniveau": b.bench_motivic_coniveau_family(),
        "motivic_atiyah": b.bench_motivic_atiyah_family(),
        "motivic_deligne": b.bench_motivic_deligne_family(),
        "motivic_residue": b.bench_motivic_residue_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
