"""SYNTHETIC radix-2 DIT FFT vs naive DFT oracle.

Cooley-Tukey on power-of-2 inputs; verified pointwise against the O(n²)
DFT and on circular convolution (conv theorem).
"""

from __future__ import annotations

import cmath
import math
import random


def fft(x: list[complex], invert: bool = False) -> list[complex]:
    n = len(x)
    if n == 1:
        return x
    ev = fft(x[0::2], invert)
    od = fft(x[1::2], invert)
    ang = (2 * math.pi / n) * (-1 if not invert else 1)
    w = 1.0 + 0j
    wn = cmath.exp(1j * ang)
    out = [0j] * n
    for k in range(n // 2):
        t = w * od[k]
        out[k] = ev[k] + t
        out[k + n // 2] = ev[k] - t
        w *= wn
    return out


def _dft(x: list[complex]) -> list[complex]:
    n = len(x)
    return [sum(x[j] * cmath.exp(-2j * math.pi * k * j / n) for j in range(n)) for k in range(n)]


def _conv_naive(a: list[float], b: list[float]) -> list[float]:
    n = len(a)
    return [sum(a[j] * b[(k - j) % n] for j in range(n)) for k in range(n)]


def bench_fft_radix2(seed: int = 20261231 + 450) -> dict[str, float]:
    rng = random.Random(seed)
    close = conv = inv = 0
    trials = 40
    for _ in range(trials):
        n = 1 << rng.randrange(3, 7)
        x = [complex(rng.uniform(-9, 9), rng.uniform(-9, 9)) for _ in range(n)]
        f_fast, f_slow = fft(x), _dft(x)
        close += int(max(abs(a - b) for a, b in zip(f_fast, f_slow, strict=True)) < 1e-8)
        # circular convolution via FFT
        a = [rng.uniform(-4, 4) for _ in range(n)]
        b = [rng.uniform(-4, 4) for _ in range(n)]
        fa, fb = fft([complex(v, 0) for v in a]), fft([complex(v, 0) for v in b])
        prod = [u * v for u, v in zip(fa, fb, strict=True)]
        c = [z.real / n for z in fft(prod, invert=True)]
        cn = _conv_naive(a, b)
        conv += int(max(abs(u - v) for u, v in zip(c, cn, strict=True)) < 1e-6)
        # inverse round-trip
        xb = [z / n for z in fft(fft(x), invert=True)]
        inv += int(max(abs(u - v) for u, v in zip(xb, x, strict=True)) < 1e-9)
    return {
        "synthetic_fft_matches_dft": float(close / trials),
        "synthetic_convolution_theorem": float(conv / trials),
        "synthetic_inverse_roundtrip": float(inv / trials),
    }
