"""Wave-730 ramification adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w730 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "grothendieck_muw": b.bench_grothendieck_muw_family(),
        "raynaud_pencil": b.bench_raynaud_pencil_family(),
        "saito_epsilon": b.bench_saito_epsilon_family(),
        "swan_conductor": b.bench_swan_conductor_family(),
        "groth_tame": b.bench_groth_tame_family(),
        "kato_swan": b.bench_kato_swan_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
