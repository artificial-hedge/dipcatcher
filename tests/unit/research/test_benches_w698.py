"""Wave-698 motivic-22 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w698 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_frobenius": b.bench_motivic_frobenius_family(),
        "motivic_cartier": b.bench_motivic_cartier_family(),
        "motivic_hodge": b.bench_motivic_hodge_family(),
        "motivic_span": b.bench_motivic_span_family(),
        "motivic_lax": b.bench_motivic_lax_family(),
        "motivic_street": b.bench_motivic_street_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
