"""Wave-766 empirical-process-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w766 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dudley_theorem": b.bench_dudley_theorem_family(),
        "varadarajan_thm": b.bench_varadarajan_thm_family(),
        "dvoretzky_thm": b.bench_dvoretzky_thm_family(),
        "vc_class": b.bench_vc_class_family(),
        "bracketing_ent": b.bench_bracketing_ent_family(),
        "bounded_lip": b.bench_bounded_lip_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
