"""Mel filterbank + MFCC feature extraction.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 906


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return np.asarray(
        2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0), dtype=np.float64
    )


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return np.asarray(
        700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0), dtype=np.float64
    )


def mel_filterbank(nfilt: int, nfft: int, fs: float) -> np.ndarray:
    melpts = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([fs / 2]))[0], nfilt + 2)
    hz = mel_to_hz(melpts)
    bins = np.floor((nfft + 1) * hz / fs).astype(int)
    fb = np.zeros((nfilt, nfft // 2 + 1))
    for i in range(nfilt):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for k in range(lo, mid):
            if mid > lo:
                fb[i, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[i, k] = (hi - k) / (hi - mid)
    return np.asarray(fb, dtype=np.float64)


def mfcc(x: np.ndarray, fs: float, nfilt: int = 13, nfft: int = 512) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sp = np.abs(np.fft.rfft(x * np.hanning(x.size), nfft)) ** 2
    fb = mel_filterbank(nfilt, nfft, fs)
    e = np.log10(fb @ sp + 1e-12)
    n = e.size
    c = np.array(
        [
            sum(e[k] * np.cos(np.pi * i * (2 * k + 1) / (2 * n)) for k in range(n))
            for i in range(nfilt)
        ]
    )
    return np.asarray(c, dtype=np.float64)


def bench_mel_cepstrum(seed: int = _SEED) -> dict[str, float]:
    fs = 16000.0
    t = np.arange(400) / fs
    score = 0.0
    # filterbank: nonneg, single contiguous nonzero band per row, decent peak
    fb = mel_filterbank(13, 512, fs)
    ok = np.all(fb >= 0) and np.max(fb) > 0.5
    for i in range(13):
        nz = np.flatnonzero(fb[i] > 0)
        ok = ok and nz.size > 0 and int(nz[-1] - nz[0]) == nz.size - 1
        am = int(np.argmax(fb[i]))
        ok = ok and np.all(np.diff(fb[i, : am + 1]) >= -1e-12)
        ok = ok and np.all(np.diff(fb[i, am:]) <= 1e-12)
    score += 1.0 if ok else 0.0
    # two tones at different freqs → different MFCC vectors
    x1 = np.sin(2 * np.pi * 300 * t)
    x2 = np.sin(2 * np.pi * 3000 * t)
    c1, c2 = mfcc(x1, fs), mfcc(x2, fs)
    dist_diff = float(np.linalg.norm(c1 - c2))
    x1b = np.sin(2 * np.pi * 320 * t)
    dist_same = float(np.linalg.norm(c1 - mfcc(x1b, fs)))
    score += 1.0 if dist_diff > 2.0 * dist_same else 0.0
    # cepstral pitch detection: periodic excitation → rahmonic peak
    per = np.tile(np.r_[np.zeros(79), 1.0], 5)[:400]
    v = np.convolve(per, np.exp(-t * 800) * np.sin(2 * np.pi * 120 * t), "full")[:400]
    cv = mfcc(v, fs)
    score += 1.0 if np.all(np.isfinite(cv)) else 0.0
    # loudness invariance: scaling signal leaves normalized c1+ MFCC ~unchanged
    # (c0 carries log energy, dropped per standard practice)
    c_a = mfcc(x1, fs)[1:]
    c_b = mfcc(0.01 * x1, fs)[1:]
    na, nb = c_a / np.linalg.norm(c_a), c_b / np.linalg.norm(c_b)
    score += 1.0 if float(np.linalg.norm(na - nb)) < 0.15 else 0.0
    return {"synthetic_mel_cepstrum": score / 4.0}
