"""Wave-731 ramification-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w731 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "higher_ramif": b.bench_higher_ramif_family(),
        "brylinski_kato": b.bench_brylinski_kato_family(),
        "log_ramification": b.bench_log_ramification_family(),
        "semi_stable_model": b.bench_semi_stable_model_family(),
        "neron_raynaud": b.bench_neron_raynaud_family(),
        "temkin_alter": b.bench_temkin_alter_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
