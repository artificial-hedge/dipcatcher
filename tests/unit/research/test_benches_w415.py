"""Wave-415 topology-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w415 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quotient_map": b.bench_quotient_map_family(),
        "open_cover": b.bench_open_cover_family(),
        "locally_compact": b.bench_locally_compact_family(),
        "homeo_top": b.bench_homeo_top_family(),
        "paracompact": b.bench_paracompact_family(),
        "partition_unity": b.bench_partition_unity_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
