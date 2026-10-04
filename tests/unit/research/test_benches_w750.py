"""Wave-750 dimer/Ising adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w750 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "smirnov_ising": b.bench_smirnov_ising_family(),
        "chelkak_ising": b.bench_chelkak_ising_family(),
        "kenyon_dimers": b.bench_kenyon_dimers_family(),
        "thurston_tiling": b.bench_thurston_tiling_family(),
        "duminil_copin2": b.bench_duminil_copin2_family(),
        "hongler_ising": b.bench_hongler_ising_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
