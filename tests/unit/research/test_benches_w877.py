"""Wave-877 transport/SPn adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w877 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "transport_sn": b.bench_transport_sn_family(),
        "discrete_ordinates": b.bench_discrete_ordinates_family(),
        "spherical_harmonics": b.bench_spherical_harmonics_family(),
        "spn_equations": b.bench_spn_equations_family(),
        "moc_transport": b.bench_moc_transport_family(),
        "pn_closure": b.bench_pn_closure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
