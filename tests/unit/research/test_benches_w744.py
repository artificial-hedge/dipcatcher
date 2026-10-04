"""Wave-744 KPZ adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w744 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kardar_parisi": b.bench_kardar_parisi_family(),
        "corwin_kpz": b.bench_corwin_kpz_family(),
        "quastel_spohn": b.bench_quastel_spohn_family(),
        "borodin_corwin": b.bench_borodin_corwin_family(),
        "amir_corwin": b.bench_amir_corwin_family(),
        "calabrese_kpz": b.bench_calabrese_kpz_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
