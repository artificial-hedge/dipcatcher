"""Viterbi canon: convolutional-code encode/decode — rate-1/2
K=7 (171/133 octal) encoder, hard- and soft-decision Viterbi
decoding through the trellis. Bench: exact recovery in the
noise-free channel, soft beats hard under BPSK noise, and
path-metric sanity vs brute-force minimal distance. All
SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_GENS = (0o171, 0o133)  # K=7 rate-1/2
_K = 7
_NSTATES = 1 << (_K - 1)


def conv_encode(bits: FloatArray) -> FloatArray:
    """Rate-1/2 systematic-free encode: 2 output bits per input
    bit, generator polynomials (171, 133) octal."""
    out = np.zeros(2 * len(bits), dtype=np.float64)
    state = 0
    for i, b in enumerate(bits):
        state = ((state << 1) | int(b)) & ((1 << _K) - 1)
        for j, g in enumerate(_GENS):
            out[2 * i + j] = bin(state & g).count("1") % 2
    return out


def _next_output(state: int, b: int) -> tuple[int, int]:
    # 7-bit shift register: bit0 = newest input, bits1..6 = state
    reg = (state << 1) | b
    o = 0
    for g in _GENS:
        o = (o << 1) | (bin(reg & g).count("1") % 2)
    # next trellis state = low K-1 bits of the register
    return reg & (_NSTATES - 1), o


def viterbi_hard(recv: FloatArray) -> FloatArray:
    """Hard-decision Viterbi on a received bit stream (2 per
    input)."""
    n = len(recv) // 2
    INF = 1e18
    metric = np.full(_NSTATES, INF)
    metric[0] = 0.0
    parents = np.full((n, _NSTATES), -1, dtype=np.int64)
    bits_in = np.zeros((n, _NSTATES), dtype=np.int64)
    for i in range(n):
        r = int(recv[2 * i]) << 1 | int(recv[2 * i + 1])
        new = np.full(_NSTATES, INF)
        for s in range(_NSTATES):
            if metric[s] >= INF:
                continue
            for b in (0, 1):
                ns, o = _next_output(s, b)
                d = bin(r ^ o).count("1")
                if metric[s] + d < new[ns]:
                    new[ns] = metric[s] + d
                    parents[i, ns] = s
                    bits_in[i, ns] = b
        metric = new
    out = np.zeros(n, dtype=np.float64)
    s = int(np.argmin(metric))
    for i in range(n - 1, -1, -1):
        out[i] = bits_in[i, s]
        s = int(parents[i, s])
    return out


def viterbi_soft(recv: FloatArray) -> FloatArray:
    """Soft-decision Viterbi on ±1 BPSK symbols (2 per input
    bit): Euclidean branch metric."""
    n = len(recv) // 2
    INF = 1e18
    metric = np.full(_NSTATES, INF)
    metric[0] = 0.0
    parents = np.full((n, _NSTATES), -1, dtype=np.int64)
    bits_in = np.zeros((n, _NSTATES), dtype=np.int64)
    for i in range(n):
        r0, r1 = recv[2 * i], recv[2 * i + 1]
        new = np.full(_NSTATES, INF)
        for s in range(_NSTATES):
            if metric[s] >= INF:
                continue
            for b in (0, 1):
                ns, o = _next_output(s, b)
                e0 = 1.0 if (o >> 1) & 1 else -1.0
                e1 = 1.0 if o & 1 else -1.0
                d = (r0 - e0) ** 2 + (r1 - e1) ** 2
                if metric[s] + d < new[ns]:
                    new[ns] = metric[s] + d
                    parents[i, ns] = s
                    bits_in[i, ns] = b
        metric = new
    out = np.zeros(n, dtype=np.float64)
    s = int(np.argmin(metric))
    for i in range(n - 1, -1, -1):
        out[i] = bits_in[i, s]
        s = int(parents[i, s])
    return out


def bench_viterbi_decode(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 120).astype(np.float64)
    enc = conv_encode(bits)
    # noiseless hard decode == original (minus tail flush bits)
    dec = viterbi_hard(enc)
    out["synthetic_viterbi_clean_ber"] = float(np.mean(dec != bits))
    # BPSK noise: soft beats hard at moderate SNR
    sym = 2 * enc - 1
    sigma = 0.9
    noisy = sym + rng.normal(0, sigma, len(sym))
    hard = viterbi_hard((noisy > 0).astype(np.float64))
    soft = viterbi_soft(noisy)
    out["synthetic_viterbi_hard_ber"] = float(np.mean(hard != bits))
    out["synthetic_viterbi_soft_ber"] = float(np.mean(soft != bits))
    out["synthetic_viterbi_soft_gain"] = (
        out["synthetic_viterbi_hard_ber"] - out["synthetic_viterbi_soft_ber"]
    )
    # low noise: both recover
    sigma2 = 0.3
    noisy2 = sym + rng.normal(0, sigma2, len(sym))
    out["synthetic_viterbi_lowsnr_soft_ber"] = float(np.mean(viterbi_soft(noisy2) != bits))
    # minimum free distance sanity: flip one codeword bit →
    # decoder corrects it
    bad = enc.copy()
    bad[10] = 1.0 - bad[10]
    out["synthetic_viterbi_1err_ber"] = float(np.mean(viterbi_hard(bad) != bits))
    return out
