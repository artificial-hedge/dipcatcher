"""Wave-515 syzygy-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w515 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "betti_series": b.bench_betti_series_family(),
        "minimal_free": b.bench_minimal_free_family(),
        "auslander_buchs": b.bench_auslander_buchs_family(),
        "serre_conj": b.bench_serre_conj_family(),
        "quillen_suslin": b.bench_quillen_suslin_family(),
        "green_koszul": b.bench_green_koszul_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
