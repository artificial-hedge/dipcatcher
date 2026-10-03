"""Wave-723 Iwasawa/Euler-system adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w723 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gross_zagier": b.bench_gross_zagier_family(),
        "kolyvagin_sys": b.bench_kolyvagin_sys_family(),
        "euler_system": b.bench_euler_system_family(),
        "iwasawa_motive": b.bench_iwasawa_motive_family(),
        "rubin_main_conj": b.bench_rubin_main_conj_family(),
        "perrin_riou": b.bench_perrin_riou_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
