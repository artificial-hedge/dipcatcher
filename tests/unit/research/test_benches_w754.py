"""Wave-754 random-cluster adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w754 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sokal_bcc": b.bench_sokal_bcc_family(),
        "caracciolo_pelissetto": b.bench_caracciolo_pelissetto_family(),
        "grimmett_rc": b.bench_grimmett_rc_family(),
        "hara_hara": b.bench_hara_hara_family(),
        "brydges_spencer": b.bench_brydges_spencer_family(),
        "glasner_aizenman": b.bench_glasner_aizenman_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
