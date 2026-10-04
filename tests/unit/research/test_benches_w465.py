"""Wave-465 analytic-geometry-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w465 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kedlaya_renorm": b.bench_kedlaya_renorm_family(),
        "dagger_groth": b.bench_dagger_groth_family(),
        "raynaud_gen": b.bench_raynaud_gen_family(),
        "weierstrass_prep": b.bench_weierstrass_prep_family(),
        "gauss_point": b.bench_gauss_point_family(),
        "affinoid_alg": b.bench_affinoid_alg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
