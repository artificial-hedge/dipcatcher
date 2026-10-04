"""Wave-827 random-series adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w827 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "three_series": b.bench_three_series_family(),
        "kolmogorov_two": b.bench_kolmogorov_two_family(),
        "ito_nisio": b.bench_ito_nisio_family(),
        "chung_series": b.bench_chung_series_family(),
        "ortega_series": b.bench_ortega_series_family(),
        "salem_zygmund": b.bench_salem_zygmund_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
