"""Wave-655 homotopy-21 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w655 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "devinatz_htpy": b.bench_devinatz_htpy_family(),
        "hopkins_smith": b.bench_hopkins_smith_family(),
        "morava_stab": b.bench_morava_stab_family(),
        "chromatic_square": b.bench_chromatic_square_family(),
        "telescope_tower": b.bench_telescope_tower_family(),
        "bo_htpy": b.bench_bo_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
