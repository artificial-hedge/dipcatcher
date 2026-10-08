"""RBJ audio-cookbook biquads + DF2T cascade (second-order sections) (SYNTHETIC).

Canonical reference: Robert Bristow-Johnson's Audio EQ Cookbook.
Provides design helpers (lowpass/highpass/bandpass/notch/peaking) and
a numerically sane cascade runner shared by the IIR module family.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _rbj(kind: str, f0: float, q: float = 0.707, gain_db: float = 0.0) -> FloatArray:
    """One biquad [b0 b1 b2 a0 a1 a2], f0 normalized to Nyquist=0.5."""
    w0 = 2 * np.pi * f0
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    a = 10 ** (gain_db / 40.0)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        aa = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        aa = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        aa = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cw, 1.0]
        aa = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peaking":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        aa = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:
        raise ValueError(f"unknown kind {kind}")
    return np.array([b[0] / aa[0], b[1] / aa[0], b[2] / aa[0], 1.0, aa[1] / aa[0], aa[2] / aa[0]])


def biquad_lowpass(f0: float, q: float = 0.707) -> FloatArray:
    return _rbj("lowpass", f0, q)


def biquad_highpass(f0: float, q: float = 0.707) -> FloatArray:
    return _rbj("highpass", f0, q)


def biquad_notch(f0: float, q: float = 8.0) -> FloatArray:
    return _rbj("notch", f0, q)


def biquad_peaking(f0: float, gain_db: float, q: float = 1.0) -> FloatArray:
    return _rbj("peaking", f0, q, gain_db)


def biquad_apply(b: FloatArray, x: FloatArray) -> FloatArray:
    """DF2T apply of a single [b0 b1 b2 a0 a1 a2] section."""
    b0, b1, b2, _, a1, a2 = b
    y = np.empty(len(x))
    w1 = w2 = 0.0
    for i in range(len(x)):
        w = x[i] - a1 * w1 - a2 * w2
        y[i] = b0 * w + b1 * w1 + b2 * w2
        w2, w1 = w1, w
    return y


def biquad_response(b: FloatArray, f: FloatArray) -> FloatArray:
    b0, b1, b2, _, a1, a2 = b
    z = np.exp(-2j * np.pi * f)
    mag: FloatArray = np.abs((b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z))
    return mag


def bench_biquad(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: lp passes DC, hp passes Nyquist, notch nulls f0,
    peaking hits gain_db; DF2T cascade filters a chirp."""
    rng = np.random.default_rng(seed)
    lp = biquad_lowpass(0.1)
    hp = biquad_highpass(0.3)
    nt = biquad_notch(0.2, q=16.0)
    pk = biquad_peaking(0.15, gain_db=6.0, q=2.0)
    f = np.array([0.0, 0.2, 0.499])
    lp_dc, lp_mid, lp_hi = biquad_response(lp, f)
    hp_lo, hp_mid, hp_hi = biquad_response(hp, f)
    notch_at = float(biquad_response(nt, np.array([0.2]))[0])
    peak_at = float(biquad_response(pk, np.array([0.15]))[0])
    n = np.arange(4096)
    x = np.sin(2 * np.pi * 0.2 * n) + 0.3 * rng.normal(size=len(n))
    y = biquad_apply(nt, x)
    residual = float(np.sqrt(np.mean(y[256:] ** 2)))
    return {
        "synthetic_biquad_lp_dc_err": float(abs(lp_dc - 1.0)),
        "synthetic_biquad_lp_hi_att": float(-20 * np.log10(lp_hi + 1e-15)),
        "synthetic_biquad_hp_hi_err": float(abs(hp_hi - 1.0)),
        "synthetic_biquad_hp_lo_att": float(-20 * np.log10(hp_lo + 1e-15)),
        "synthetic_biquad_notch_att": float(-20 * np.log10(notch_at + 1e-15)),
        "synthetic_biquad_peak_gain_db": float(20 * np.log10(peak_at + 1e-15)),
        "synthetic_biquad_notch_residual": residual,
    }
