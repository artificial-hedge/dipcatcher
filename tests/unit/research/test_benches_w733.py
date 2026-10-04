"""Wave-733 Hall-algebra-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w733 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "green_hall": b.bench_green_hall_family(),
        "bridgeland_hall": b.bench_bridgeland_hall_family(),
        "kontsevich_soibelman": b.bench_kontsevich_soibelman_family(),
        "mozgovoy_hall": b.bench_mozgovoy_hall_family(),
        "morita_hall": b.bench_morita_hall_family(),
        "calaque_hall": b.bench_calaque_hall_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
