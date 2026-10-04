"""Wave-388 homological-algebra-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w388 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_functor": b.bench_derived_functor_family(),
        "ext_compute": b.bench_ext_compute_family(),
        "tor_compute": b.bench_tor_compute_family(),
        "spectral_seq": b.bench_spectral_seq_family(),
        "koszul_homology": b.bench_koszul_homology_family(),
        "mapping_degree": b.bench_mapping_degree_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
