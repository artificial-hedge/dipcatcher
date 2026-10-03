"""Wave-811 point-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w811 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cambrian_pp": b.bench_cambrian_pp_family(),
        "papangelou": b.bench_papangelou_family(),
        "gneding_metric": b.bench_gneding_metric_family(),
        "void_prob": b.bench_void_prob_family(),
        "j_function": b.bench_j_function_family(),
        "ergodic_pp": b.bench_ergodic_pp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
