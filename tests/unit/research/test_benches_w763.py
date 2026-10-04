"""Wave-763 mixing/urn adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w763 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "polya_urn": b.bench_polya_urn_family(),
        "hopf_chain": b.bench_hopf_chain_family(),
        "boneschi_boal": b.bench_boneschi_boal_family(),
        "bradley_mixing": b.bench_bradley_mixing_family(),
        "rosenthal_mom": b.bench_rosenthal_mom_family(),
        "ibagimov_mixing": b.bench_ibagimov_mixing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
