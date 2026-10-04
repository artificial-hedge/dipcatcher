"""Wave-745 KPZ-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w745 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dotsenko_kpz": b.bench_dotsenko_kpz_family(),
        "hairer_kpz": b.bench_hairer_kpz_family(),
        "bernard_nicola": b.bench_bernard_nicola_family(),
        "imamura_sasamoto": b.bench_imamura_sasamoto_family(),
        "tracy_widom_kpz": b.bench_tracy_widom_kpz_family(),
        "spohn_kpz": b.bench_spohn_kpz_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
