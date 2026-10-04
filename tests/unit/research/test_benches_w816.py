"""Wave-816 Markov adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w816 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hunt_process": b.bench_hunt_process_family(),
        "cadlag_markov": b.bench_cadlag_markov_family(),
        "transition_semigroup": b.bench_transition_semigroup_family(),
        "resolvent_markov": b.bench_resolvent_markov_family(),
        "generator_markov": b.bench_generator_markov_family(),
        "characteristic_markov": b.bench_characteristic_markov_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
