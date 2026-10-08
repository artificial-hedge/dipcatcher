"""Continuous wavelet transform with Morlet + ridge (instantaneous freq) (SYNTHETIC).

CWT via FFT convolution of the Morlet family; the per-time argmax scale
traces the chirp's instantaneous frequency. Bench: ridge f0(t) vs the
planted linear f(t) law — median relative error.
"""

import numpy as np

from quant_fund.models._sig3_synth import FS, chirp


def _morlet(f: float, t: np.ndarray) -> np.ndarray:
    sig = 6.0 / (2 * np.pi * f)
    w = np.exp(2j * np.pi * f * t) * np.exp(-(t**2) / (2 * sig**2))
    return w - w.mean()


def cwt_ridge_path(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros((len(freqs), n))
    xf = np.fft.fft(x, n=2 * n)
    t = np.arange(-n, n) / FS
    for i, f in enumerate(freqs):
        w = np.fft.ifftshift(_morlet(f, t))  # wavelet center at index 0
        conv = np.fft.ifft(xf * np.conj(np.fft.fft(w, n=2 * n)))
        out[i] = np.abs(conv[:n])
    return out


def bench_cwt_ridge(seed: int = 4801) -> dict[str, float]:
    x = chirp(seed)
    freqs = np.linspace(150, 900, 60)
    spec = cwt_ridge_path(x, freqs)
    ridge = freqs[np.argmax(spec, axis=0)]
    t = np.arange(len(x)) / FS
    true_f = 200.0 + 600.0 * t / (len(x) / FS)
    edge = len(x) // 10
    rel = np.abs(ridge[edge:-edge] - true_f[edge:-edge]) / true_f[edge:-edge]
    return {
        "synthetic_cwt_med_err": float(np.median(rel)),
        "synthetic_cwt_p90_err": float(np.quantile(rel, 0.9)),
        "synthetic_cwt_ridge_start": float(ridge[edge]),
        "synthetic_cwt_ridge_end": float(ridge[-edge]),
        "synthetic_cwt_true_end": float(true_f[-edge]),
    }
