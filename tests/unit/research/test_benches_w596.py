"""Wave-596 p-adic-Hodge adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w596 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "breuil_mod": b.bench_breuil_mod_family(),
        "kisin_mod": b.bench_kisin_mod_family(),
        "galois_lattice": b.bench_galois_lattice_family(),
        "padic_hodge": b.bench_padic_hodge_family(),
        "finite_height": b.bench_finite_height_family(),
        "etale_phi": b.bench_etale_phi_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
