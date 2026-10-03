"""Adapter tests for wave-286 security-defensive canon benches."""

from quant_fund.research.benches_w286 import (
    bench_beacon_detect_family,
    bench_cred_stuffing_family,
    bench_entropy_dns_family,
    bench_exfil_zscore_family,
    bench_impossible_travel_family,
    bench_sig_score_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_beacon_detect_family,
        bench_entropy_dns_family,
        bench_cred_stuffing_family,
        bench_impossible_travel_family,
        bench_exfil_zscore_family,
        bench_sig_score_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
