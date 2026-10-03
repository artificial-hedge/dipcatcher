"""Polar code (16,8): butterfly encoding + successive-cancellation decode.

Channel polarization via the BEC Bhattacharyya recursion on a design
BSC; the 8 most reliable bit-channels carry info, the rest are frozen
to zero. SC decoding uses the min-sum f-rule and box-plus g-rule.
Bench compares SC BER vs uncoded transmission on the shared channel.
"""

import numpy as np

from quant_fund.models._code_synth import POLAR_DESIGN, POLAR_K, POLAR_N, bsc, msg_bits


def _polarize(z: float, n: int) -> np.ndarray:
    zc = np.array([z])
    for _ in range(int(np.log2(n))):
        zc = np.concatenate([zc * (2 - zc), zc**2])
    return zc


def _encode(u: np.ndarray, info: np.ndarray) -> np.ndarray:
    n = len(u)
    x = u.copy()
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
        step *= 2
    return x


def _sc_decode(y_llr: np.ndarray, info: np.ndarray) -> np.ndarray:
    """Successive-cancellation decode; returns decoded u vector."""
    n = len(y_llr)
    bits = np.zeros(n, dtype=int)
    frozen = ~info

    def dec(llr: np.ndarray, lo: int, hi: int) -> None:
        size = hi - lo
        if size == 1:
            bits[lo] = 0 if frozen[lo] or llr[0] > 0 else 1
            return
        half = size // 2
        f = (
            np.sign(llr[:half])
            * np.sign(llr[half:])
            * np.minimum(np.abs(llr[:half]), np.abs(llr[half:]))
        )
        dec(f, lo, lo + half)
        g = llr[half:] + (1 - 2 * bits[lo : lo + half]) * llr[:half]
        dec(g, lo + half, hi)

    dec(y_llr, 0, n)
    return bits


def bench_polar_code(seed: int = 5005, p: float = 0.11) -> dict[str, float]:
    z = _polarize(POLAR_DESIGN, POLAR_N)
    order = np.argsort(z)  # most reliable first
    info = np.zeros(POLAR_N, dtype=bool)
    info[order[:POLAR_K]] = True
    m = msg_bits(seed, POLAR_K)
    u = np.zeros(POLAR_N, dtype=int)
    u[info] = m
    c = _encode(u, info)
    y = bsc(c, p, seed + 1)
    y_llr = np.where(y == 0, 4.0, -4.0)
    dec = _sc_decode(y_llr, info)
    unc = bsc(m, p, seed + 2)
    return {
        "synthetic_polar_ber": float(np.mean(dec != u)),
        "synthetic_polar_msg_ber": float(np.mean(dec[info] != m)),
        "synthetic_polar_uncoded_ber": float(np.mean(unc != m)),
        "synthetic_polar_gain": float(np.mean(unc != m) - np.mean(dec[info] != m)),
        "synthetic_polar_info": float(np.sum(info)),
    }
