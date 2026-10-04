"""Wave-896 RK/IVP adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w896 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fehlberg_rk": b.bench_fehlberg_rk_family(),
        "dormand_prince": b.bench_dormand_prince_family(),
        "cash_karp": b.bench_cash_karp_family(),
        "bogacki_shampine": b.bench_bogacki_shampine_family(),
        "backward_euler": b.bench_backward_euler_family(),
        "predictor_corrector": b.bench_predictor_corrector_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
