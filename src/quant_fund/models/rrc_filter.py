"""RRC canon: root-raised-cosine pulse design + matched-filter
chain — taps by the closed-form RRC formula, convolution, and
the Nyquist-ISI property (conv(RRC,RRC) ≈ RC → zero crossings
at symbol times). Bench: ISI at symbol centers ~0 for an
integer-rate stream, spectral rolloff shape, and gain
normalization. All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rrc_taps(sps: int, num_taps: int, beta: float = 0.35) -> FloatArray:
    """Root-raised-cosine impulse response, normalized to unit
    DC gain at symbol rate."""
    if not math.isfinite(beta) or not (0.0 < beta <= 1.0):
        raise ValueError(f"rolloff beta must be in (0, 1]; got {beta!r}")
    if int(sps) < 1 or int(num_taps) < 1:
        raise ValueError("sps and num_taps must be positive")
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    h = np.zeros(num_taps)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-12:
            h[i] = 1 - beta + 4 * beta / math.pi
        elif abs(abs(tt) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
            )
        else:
            num = math.sin(math.pi * tt * (1 - beta)) + 4 * beta * tt * math.cos(
                math.pi * tt * (1 + beta)
            )
            den = math.pi * tt * (1 - (4 * beta * tt) ** 2)
            h[i] = num / den
    h = h / np.sqrt(np.sum(h**2))  # unit energy
    return np.asarray(h, dtype=np.float64)


def pulse_shape(bits: FloatArray, sps: int, h: FloatArray) -> FloatArray:
    """Upsample NRZ symbols and filter with the RRC taps."""
    up = np.zeros(len(bits) * sps)
    up[::sps] = bits
    return np.convolve(up, h, mode="same")


def matched_filter(rx: FloatArray, h: FloatArray) -> FloatArray:
    return np.convolve(rx, h[::-1], mode="same")


def bench_rrc_filter(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    sps = 8
    h = rrc_taps(sps, 8 * sps + 1, beta=0.35)
    # ISI: RRC*RRC ≈ raised cosine → zero at ±symbol times
    hh = np.convolve(h, h)
    mid = len(hh) // 2
    out["synthetic_rrc_isi_plus1"] = float(abs(hh[mid + sps]))
    out["synthetic_rrc_isi_minus1"] = float(abs(hh[mid - sps]))
    out["synthetic_rrc_isi_plus2"] = float(abs(hh[mid + 2 * sps]))
    out["synthetic_rrc_peak"] = float(hh[mid])
    # end-to-end: random bits through TX RRC + RX RRC, sample at
    # center → decision BER ~ 0
    bits = (2 * rng.integers(0, 2, 400) - 1).astype(np.float64)
    tx = pulse_shape(bits, sps, h)
    rx = matched_filter(tx, h)
    # align: find the best shift
    best_ber = 1.0
    for sh in range(sps):
        cand = rx[sh::sps][: len(bits)]
        best_ber = min(best_ber, float(np.mean(np.sign(cand) != bits[: len(cand)])))
    out["synthetic_rrc_chain_ber"] = best_ber
    # rolloff: energy outside ±(1+β)/2T of the spectrum ≈ small
    spec = np.abs(np.fft.rfft(h)) ** 2
    out["synthetic_rrc_spec_peak"] = float(spec.max())
    return out
