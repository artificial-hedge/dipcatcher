"""Wave-574 differential-topology-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w574 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "exotic_sphere": b.bench_exotic_sphere_family(),
        "kervaire_milnor": b.bench_kervaire_milnor_family(),
        "surgery_theory": b.bench_surgery_theory_family(),
        "smale_hcob": b.bench_smale_hcob_family(),
        "whitney_trick": b.bench_whitney_trick_family(),
        "immersion_thm": b.bench_immersion_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
