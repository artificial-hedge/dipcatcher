"""Wave-815 excursion adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w815 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "excursion_proc": b.bench_excursion_proc_family(),
        "inverse_local": b.bench_inverse_local_family(),
        "ray_knight": b.bench_ray_knight_family(),
        "knight_theorem": b.bench_knight_theorem_family(),
        "mazza_yor": b.bench_mazza_yor_family(),
        "pitman_thm": b.bench_pitman_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
