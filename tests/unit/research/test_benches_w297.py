"""Adapter tests for wave-297 post-quantum crypto canon benches."""

from quant_fund.research.benches_w297 import (
    bench_dilithium_sig_family,
    bench_frodokem_family,
    bench_kyber_kem_family,
    bench_ntt_ring_family,
    bench_sphincs_sig_family,
    bench_xmss_sig_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_ntt_ring_family,
        bench_kyber_kem_family,
        bench_dilithium_sig_family,
        bench_frodokem_family,
        bench_xmss_sig_family,
        bench_sphincs_sig_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
