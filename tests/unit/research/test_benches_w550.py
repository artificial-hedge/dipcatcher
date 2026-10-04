"""Wave-550 characteristic-classes adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w550 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chern_class": b.bench_chern_class_family(),
        "pontryagin_class": b.bench_pontryagin_class_family(),
        "euler_class": b.bench_euler_class_family(),
        "todd_genus": b.bench_todd_genus_family(),
        "chern_character": b.bench_chern_character_family(),
        "hirzebruch_sig": b.bench_hirzebruch_sig_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
