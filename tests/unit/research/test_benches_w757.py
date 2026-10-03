"""Wave-757 CLE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w757 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sheffield_werner_cle": b.bench_sheffield_werner_cle_family(),
        "miller_watson_cle": b.bench_miller_watson_cle_family(),
        "camia_newman": b.bench_camia_newman_family(),
        "dubedat_cle": b.bench_dubedat_cle_family(),
        "kemppainen_werner": b.bench_kemppainen_werner_family(),
        "rivera_cle": b.bench_rivera_cle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
