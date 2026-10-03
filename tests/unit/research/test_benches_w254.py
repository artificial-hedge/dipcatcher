"""Wave-254 adapter tests."""

from quant_fund.research.benches_w254 import (
    bench_aead_etm_family,
    bench_cbc_padding_family,
    bench_hmac_construct_family,
    bench_merkle_damgard_family,
    bench_pbkdf2_kdf_family,
    bench_tls_handshake_family,
)


def test_bench_tls_handshake_family():
    out = bench_tls_handshake_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_hmac_construct_family():
    out = bench_hmac_construct_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_aead_etm_family():
    out = bench_aead_etm_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_merkle_damgard_family():
    out = bench_merkle_damgard_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_cbc_padding_family():
    out = bench_cbc_padding_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_pbkdf2_kdf_family():
    out = bench_pbkdf2_kdf_family()
    assert out and all(k.startswith("synthetic_") for k in out)
