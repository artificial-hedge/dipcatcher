"""GPS C/A Gold codes — 1023-chip PRN spreading sequences (SYNTHETIC).

Two 10-bit maximal-length LFSRs: G1 = x^10 + x^3 + 1 and
G2 = x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1. A satellite's PRN code
is G1 delayed XOR (G2 tapped at a PRN-specific phase pair). Includes
circular-correlation code-phase acquisition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_G1_TAPS = (10, 3)
_G2_TAPS = (10, 9, 8, 6, 3, 2)
# IS-GPS-200 PRN → G2 tap pair
_PRN_TAPS = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
    33: (5, 10),
    34: (4, 10),
    35: (1, 7),
    36: (2, 8),
    37: (4, 10),
}


def _lfsr(reg: list[int], taps: tuple[int, ...]) -> int:
    out = reg[-1]
    fb = 0
    for t in taps:
        fb ^= reg[t - 1]
    reg.pop()
    reg.insert(0, fb)
    return out


def gold_code(prn: int, n: int = 1023) -> FloatArray:
    """PRN spreading code as ±1 chips (length 1023)."""
    s1, s2 = _PRN_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    chips = np.empty(n)
    for i in range(n):
        o1 = _lfsr(g1, _G1_TAPS)
        out2 = g2[s1 - 1] ^ g2[s2 - 1]
        _lfsr(g2, _G2_TAPS)
        chips[i] = 2.0 * (o1 ^ out2) - 1.0
    return chips


def acquire(signal: FloatArray, prn: int) -> tuple[int, float]:
    """Circular correlation code-phase acquisition.

    Returns (chip_offset, peak_to_mean_ratio).
    """
    x = np.asarray(signal)
    code = gold_code(prn, len(x))
    n = len(x)
    corr = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(code))).real
    peak = int(np.argmax(corr))
    off_peak = np.sort(corr)[-2] if n > 1 else 0.0
    return peak, float(corr[peak] / max(off_peak, 1e-12))


def bench_gold_code(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: acquire a PRN code shifted by a known chip offset; the
    37-code family is near-orthogonal."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    code = gold_code(1)
    shift = 373
    sig = np.roll(code, shift) + rng.normal(0, 0.05, code.size)
    est, pmr = acquire(sig, 1)
    out["synthetic_gold_offset_err"] = float(abs(est - shift))
    out["synthetic_gold_peak_ratio"] = float(pmr)
    # orthogonality: worst cross-correlation across PRNs
    worst = 0.0
    for p2 in range(2, 15):
        c2 = np.roll(gold_code(p2), shift)
        cc = np.abs(np.sum(c2 * code)) / code.size
        worst = max(worst, float(cc))
    out["synthetic_gold_crosscorr"] = worst
    ac = float(np.sum(code * code)) / code.size
    out["synthetic_gold_autocorr_peak"] = ac
    out["synthetic_gold_detected"] = float(out["synthetic_gold_offset_err"] == 0)
    return out


if __name__ == "__main__":
    print(bench_gold_code())
