"""Wave-838 discrete-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w838 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "radon_theorem": b.bench_radon_theorem_family(),
        "caratheodory_thm": b.bench_caratheodory_thm_family(),
        "farkas_lemma": b.bench_farkas_lemma_family(),
        "separation_thm": b.bench_separation_thm_family(),
        "lattice_point": b.bench_lattice_point_family(),
        "tverberg_thm": b.bench_tverberg_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
