"""Wave-492 log-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w492 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "log_structure": b.bench_log_structure_family(),
        "kato_fontaine": b.bench_kato_fontaine_family(),
        "log_smooth": b.bench_log_smooth_family(),
        "log_etale": b.bench_log_etale_family(),
        "log_derham": b.bench_log_derham_family(),
        "log_crystalline": b.bench_log_crystalline_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
