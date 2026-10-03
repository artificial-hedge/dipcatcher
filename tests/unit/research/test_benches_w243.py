"""Wave-243 adapter tests."""

from quant_fund.research.benches_w243 import (
    bench_bignum_family,
    bench_fft_radix2_family,
    bench_int_sqrt_family,
    bench_karatsuba_family,
    bench_ntt_family,
    bench_strassen_family,
)


def test_bench_bignum_family():
    out = bench_bignum_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_fft_radix2_family():
    out = bench_fft_radix2_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_int_sqrt_family():
    out = bench_int_sqrt_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_karatsuba_family():
    out = bench_karatsuba_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ntt_family():
    out = bench_ntt_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_strassen_family():
    out = bench_strassen_family()
    assert out and all(k.startswith("synthetic_") for k in out)
