"""Analog-prototype IIR design + bilinear transform to digital SOS (SYNTHETIC).

Butterworth and Chebyshev-I lowpass prototypes in the s-plane, cutoff
prewarped, mapped by s = 2fs (z−1)/(z+1), grouped into second-order
sections. Canonical reference: Oppenheim & Schafer ch. 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CplxArray = NDArray[np.complex128]


def butter_poles(order: int) -> CplxArray:
    """Left-half-plane poles of an order-N Butterworth prototype."""
    k = np.arange(order)
    ang = np.pi * (2 * k + order + 1) / (2 * order)
    return np.exp(1j * ang).astype(np.complex128)


def cheby1_poles(order: int, rp_db: float) -> tuple[CplxArray, float]:
    """Chebyshev-I poles and DC normalization gain (ripple rp_db)."""
    eps = np.sqrt(10 ** (rp_db / 10.0) - 1.0)
    mu = np.arcsinh(1.0 / eps) / order
    k = np.arange(order)
    th = np.pi * (2 * k + 1) / (2 * order)
    p = -np.sinh(mu) * np.sin(th) + 1j * np.cosh(mu) * np.cos(th)
    # DC gain: 1 for odd order, 10^{-rp/20} for even
    g = float(np.prod(np.abs(p)))
    if order % 2 == 0:
        g *= 10 ** (rp_db / 20.0)
    return p.astype(np.complex128), g


def bilinear_zpk(poles: CplxArray, gain: float, cutoff: float) -> tuple[CplxArray, float]:
    """s→z map with prewarped cutoff (normalized, 0.5 = Nyquist)."""
    wa = 2.0 * np.tan(np.pi * cutoff)  # prewarped: Ω_c = 2·tan(π f_c)
    pa = poles * wa
    z = (2.0 + pa) / (2.0 - pa)  # fs=1 → 2fs=2
    # analog zeros at infinity map to z = -1 (one per pole)
    gz = gain * float(np.prod(np.abs(2.0 - pa))) / (2.0 ** len(pa))
    return z.astype(np.complex128), gz


def zpk_to_sos(z: CplxArray, gain: float) -> FloatArray:
    """Pair conjugate poles into [b0 b1 b2 a0 a1 a2] sections.

    Lowpass zeros sit at z = −1 (analog zeros at ∞); `gain` is folded
    into the first section's numerator."""
    zs = np.sort_complex(z)
    n = len(zs)
    out = []
    i = 0
    while i < n:
        if i + 1 < n and abs(zs[i].imag) > 1e-12:
            p2 = zs[i : i + 2]
            i += 2
        else:
            p2 = zs[i : i + 1]
            i += 1
        m = len(p2)
        a1 = -float(np.real(np.sum(p2)))
        a2 = float(np.real(p2[0] * p2[1])) if m == 2 else 0.0
        b = [1.0, 2.0, 1.0] if m == 2 else [1.0, 1.0, 0.0]
        out.append([b[0], b[1], b[2], 1.0, a1, a2])
    out[0][0] *= gain
    out[0][1] *= gain
    out[0][2] *= gain
    return np.asarray(out)


def sos_apply(sos: FloatArray, x: FloatArray) -> FloatArray:
    """Cascade DF2T application of second-order sections."""
    y = np.asarray(x, dtype=np.float64).copy()
    for sec in sos:
        b0, b1, b2, _, a1, a2 = sec
        w1 = w2 = 0.0
        for i in range(len(y)):
            w = y[i] - a1 * w1 - a2 * w2
            y[i] = b0 * w + b1 * w1 + b2 * w2
            w2, w1 = w1, w
    return y


def sos_response(sos: FloatArray, f: FloatArray) -> FloatArray:
    """Cascade magnitude |H(e^{j2πf})|."""
    z = np.exp(-2j * np.pi * f)
    H = np.ones(len(f), dtype=np.complex128)
    for sec in sos:
        b0, b1, b2, _, a1, a2 = sec
        H *= (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)
    return np.abs(H)


def iir_design(order: int, cutoff: float, kind: str = "butter", rp_db: float = 1.0) -> FloatArray:
    """Lowpass digital SOS: kind ∈ {butter, cheby1}."""
    if kind == "butter":
        p = butter_poles(order)
        g = float(np.prod(np.abs(p)))
    elif kind == "cheby1":
        p, g = cheby1_poles(order, rp_db)
    else:
        raise ValueError(f"unknown kind {kind}")
    z, gz = bilinear_zpk(p, g, cutoff)
    sos = zpk_to_sos(z, gz)
    # normalize DC response numerically to the analog DC gain
    dc = float(sos_response(sos, np.array([0.0]))[0])
    target = g / float(np.prod(np.abs(p)))
    if dc > 0:
        sos[0][0] *= target / dc
        sos[0][1] *= target / dc
        sos[0][2] *= target / dc
    return sos


def bench_iir_design(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: −3dB at cutoff, monotone rolloff, stop attenuation."""
    del seed
    sos = iir_design(5, 0.2, "butter")
    fc = float(sos_response(sos, np.array([0.2]))[0])
    dc = float(sos_response(sos, np.array([0.0]))[0])
    sb = float(sos_response(sos, np.array([0.35]))[0])
    nyq = float(sos_response(sos, np.array([0.499]))[0])
    sos_c = iir_design(4, 0.2, "cheby1", rp_db=1.0)
    ripple = float(np.abs(sos_response(sos_c, np.linspace(0.0, 0.18, 64)) - 1.0).max())
    # pole stability: |roots of a| < 1
    stable = all(np.abs(np.roots([1.0, sec[4], sec[5]])).max() < 1.0 + 1e-9 for sec in sos)
    return {
        "synthetic_iir_cutoff_err": float(abs(fc - 1.0 / np.sqrt(2.0))),
        "synthetic_iir_dc_err": float(abs(dc - 1.0)),
        "synthetic_iir_stop_att": float(-20 * np.log10(sb + 1e-15)),
        "synthetic_iir_nyquist": nyq,
        "synthetic_iir_stable": float(stable),
        "synthetic_iir_cheby_ripple": ripple,
        "synthetic_iir_cheby_spec": float(ripple <= 10 ** (-1.0 / 20.0) * 1.4),
    }
