"""Wave-751 dimer-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w751 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kassel_kenyon": b.bench_kassel_kenyon_family(),
        "ciucu_dimers": b.bench_ciucu_dimers_family(),
        "karl_dimers": b.bench_karl_dimers_family(),
        "petrov_dimer": b.bench_petrov_dimer_family(),
        "durfee_arctic": b.bench_durfee_arctic_family(),
        "cohn_elkies": b.bench_cohn_elkies_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
