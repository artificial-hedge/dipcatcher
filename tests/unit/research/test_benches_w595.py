"""Wave-595 crystalline adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w595 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "crys_cohom": b.bench_crys_cohom_family(),
        "syntomic": b.bench_syntomic_family(),
        "divided_power": b.bench_divided_power_family(),
        "pd_envelope": b.bench_pd_envelope_family(),
        "nygaard_filt": b.bench_nygaard_filt_family(),
        "conjugate_fil": b.bench_conjugate_fil_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
