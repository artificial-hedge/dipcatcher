"""LPC formant estimation via the autocorrelation/Levinson method (SYNTHETIC).

All-pole model order 12; formants = peaks of the spectral envelope
|1/A(f)|. Bench: two lowest envelope peaks on a synthetic vowel-like
signal (resonances planted at 500 and 1500 Hz via biquad filters).
"""

import numpy as np

from quant_fund.models._sig3_synth import FS, voiced


def _biquad_resonator(x: np.ndarray, f: float, bw: float = 80.0) -> np.ndarray:
    r = np.exp(-np.pi * bw / FS)
    a = [1.0, -2 * r * np.cos(2 * np.pi * f / FS), r * r]
    y = np.zeros(len(x))
    for n in range(2, len(x)):
        y[n] = x[n] - a[1] * y[n - 1] - a[2] * y[n - 2]
    return y


def _lpc(x: np.ndarray, p: int) -> np.ndarray:
    r = np.correlate(x, x, "full")[len(x) - 1 : len(x) - 1 + p + 1]
    # Levinson-Durbin
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, p + 1):
        acc = r[i] + np.sum(a[1:i] * r[i - 1 : 0 : -1])
        k = -acc / e
        a[1 : i + 1] += k * a[i - 1 :: -1][:i]
        e *= 1 - k * k
    return a


def bench_lpc_formant(seed: int = 4809) -> dict[str, float]:
    src = voiced(seed)
    y = _biquad_resonator(_biquad_resonator(src, 500.0), 1500.0)
    a = _lpc(y, 12)
    # spectral envelope |1/A(f)| — formants = distinct local peaks
    w = np.linspace(0, np.pi, 2048)
    resp = np.abs(1.0 / np.polyval(a[::-1], np.exp(1j * w)))
    peaks = []
    for i in range(1, len(resp) - 1):
        if resp[i] > resp[i - 1] and resp[i] >= resp[i + 1]:
            peaks.append(i)
    peaks.sort(key=lambda i: -resp[i])
    fpk = sorted(w[peaks[:2]] * FS / (2 * np.pi)) if len(peaks) >= 2 else [0.0, 0.0]
    f1, f2 = float(fpk[0]), float(fpk[1])
    return {
        "synthetic_lpc_f1": f1,
        "synthetic_lpc_f2": f2,
        "synthetic_lpc_f1_err": abs(f1 - 500.0),
        "synthetic_lpc_f2_err": abs(f2 - 1500.0),
        "synthetic_lpc_n_peaks": float(len(peaks)),
    }
