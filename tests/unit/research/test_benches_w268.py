"""Wave-268 adapter bench tests."""

from quant_fund.research.benches_w268 import (
    bench_chacha_stream_family,
    bench_elgamal_enc_family,
    bench_fiat_shamir_family,
    bench_ot_12_family,
    bench_paillier_he_family,
    bench_poly1305_mac_family,
)

FAMS = [
    bench_elgamal_enc_family,
    bench_paillier_he_family,
    bench_fiat_shamir_family,
    bench_ot_12_family,
    bench_chacha_stream_family,
    bench_poly1305_mac_family,
]


def test_wave268_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave268_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
