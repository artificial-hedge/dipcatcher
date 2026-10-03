"""Wave-619 homotopy-14 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w619 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "unstable_htpy": b.bench_unstable_htpy_family(),
        "tame_htpy": b.bench_tame_htpy_family(),
        "devissage_ss": b.bench_devissage_ss_family(),
        "andersen_lannes": b.bench_andersen_lannes_family(),
        "chromatic_hopkins": b.bench_chromatic_hopkins_family(),
        "thick_spectrum": b.bench_thick_spectrum_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
