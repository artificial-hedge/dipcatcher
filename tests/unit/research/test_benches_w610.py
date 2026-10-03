"""Wave-610 deformations-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w610 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "schlessinger2": b.bench_schlessinger2_family(),
        "prorepresent": b.bench_prorepresent_family(),
        "versal_def": b.bench_versal_def_family(),
        "semiuniversal": b.bench_semiuniversal_family(),
        "first_order": b.bench_first_order_family(),
        "obstruction_def": b.bench_obstruction_def_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
