"""Wave-554 DT/GW-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w554 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kontsevich_mgn": b.bench_kontsevich_mgn_family(),
        "gw_descendant": b.bench_gw_descendant_family(),
        "donaldson_thomas": b.bench_donaldson_thomas_family(),
        "pandharipande_thomas": b.bench_pandharipande_thomas_family(),
        "gopakumar_vafa": b.bench_gopakumar_vafa_family(),
        "mnop_conj": b.bench_mnop_conj_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
