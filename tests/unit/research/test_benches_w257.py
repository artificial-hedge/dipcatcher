"""Wave-257 adapter tests."""

from quant_fund.research.benches_w257 import (
    bench_bfv_fhe_family,
    bench_chaum_pedersen_family,
    bench_lwe_kex_family,
    bench_ntru_toy_family,
    bench_sigma_or_proof_family,
    bench_sis_hash_family,
)


def test_bench_lwe_kex_family():
    out = bench_lwe_kex_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ntru_toy_family():
    out = bench_ntru_toy_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_bfv_fhe_family():
    out = bench_bfv_fhe_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_sis_hash_family():
    out = bench_sis_hash_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_sigma_or_proof_family():
    out = bench_sigma_or_proof_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_chaum_pedersen_family():
    out = bench_chaum_pedersen_family()
    assert out and all(k.startswith("synthetic_") for k in out)
