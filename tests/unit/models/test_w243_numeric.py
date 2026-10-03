"""Wave-243 numeric-2 canon tests."""

from quant_fund.models.bignum import _from_digits, _to_digits, add, bench_bignum, mul
from quant_fund.models.fft_radix2 import bench_fft_radix2, fft
from quant_fund.models.int_sqrt import bench_int_sqrt, isqrt, kth_root
from quant_fund.models.karatsuba import bench_karatsuba, karatsuba
from quant_fund.models.ntt import bench_ntt, ntt_mul
from quant_fund.models.strassen import bench_strassen, strassen


def test_fft_basic():
    x = [1 + 0j, 0j, 0j, 0j]
    assert all(abs(v - 1) < 1e-9 for v in fft(x))


def test_fft_bench():
    assert bench_fft_radix2()["synthetic_fft_matches_dft"] == 1.0


def test_ntt_mul_small():
    assert ntt_mul([1, 2], [3, 4]) == [3, 10, 8]


def test_ntt_bench():
    assert bench_ntt()["synthetic_ntt_mul_exact"] == 1.0


def test_karatsuba_small():
    assert karatsuba(123456789, 987654321) == 123456789 * 987654321


def test_karatsuba_bench():
    assert bench_karatsuba()["synthetic_exact_vs_builtin"] == 1.0


def test_strassen_2x2():
    a, b = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    assert strassen(a, b) == [[19, 22], [43, 50]]


def test_strassen_bench():
    assert bench_strassen()["synthetic_exact_vs_naive"] == 1.0


def test_isqrt_known():
    assert isqrt(99) == 9 and isqrt(100) == 10 and kth_root(63, 2) == 7


def test_int_sqrt_bench():
    assert bench_int_sqrt()["synthetic_isqrt_exact"] == 1.0


def test_bignum_small():
    a, b = _to_digits(12345678901234567890), _to_digits(987654321)
    assert _from_digits(add(a, b)) == 12345678901234567890 + 987654321
    assert _from_digits(mul(a, b)) == 12345678901234567890 * 987654321


def test_bignum_bench():
    assert bench_bignum()["synthetic_mul_exact"] == 1.0
