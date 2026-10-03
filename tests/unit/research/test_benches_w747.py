"""Wave-747 ASEP-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w747 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bertini_giacomin": b.bench_bertini_giacomin_family(),
        "gardina_asym": b.bench_gardina_asym_family(),
        "schutz_tasep": b.bench_schutz_tasep_family(),
        "balazs_seppalainen": b.bench_balazs_seppalainen_family(),
        "quastel_valko": b.bench_quastel_valko_family(),
        "timar_tasep": b.bench_timar_tasep_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
