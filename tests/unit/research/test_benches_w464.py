"""Wave-464 chromatic-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w464 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "morava_e": b.bench_morava_e_family(),
        "tmf_spectrum": b.bench_tmf_spectrum_family(),
        "k_n_local": b.bench_k_n_local_family(),
        "chromatic_conv": b.bench_chromatic_conv_family(),
        "nilpotence_dev": b.bench_nilpotence_dev_family(),
        "telescopic": b.bench_telescopic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
