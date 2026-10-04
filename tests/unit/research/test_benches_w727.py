"""Wave-727 Galois-deformation-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w727 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "galdef_ring": b.bench_galdef_ring_family(),
        "patching_arg": b.bench_patching_arg_family(),
        "taylor_wiles": b.bench_taylor_wiles_family(),
        "breuil_meizard": b.bench_breuil_meizard_family(),
        "gee_kisin": b.bench_gee_kisin_family(),
        "caruso_lebaron": b.bench_caruso_lebaron_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
