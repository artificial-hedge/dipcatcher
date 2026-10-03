"""STFT + perfect-reconstruction iSTFT via overlap-add."""

import numpy as np

_SEED = 20261231 + 746


def stft(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    w = np.hanning(win)
    frames = []
    for s in range(0, len(x) - win + 1, hop):
        frames.append(np.fft.rfft(x[s : s + win] * w))
    return np.array(frames)


def istft(spec: np.ndarray, win: int, hop: int, n: int) -> np.ndarray:
    w = np.hanning(win)
    out = np.zeros(n)
    norm = np.zeros(n)
    for i, f in enumerate(spec):
        seg = np.fft.irfft(f, win) * w
        s = i * hop
        out[s : s + win] += seg
        norm[s : s + win] += w**2
    norm[norm < 1e-8] = 1.0
    return out / norm


def bench_stft_istft(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n, win, hop = 256, 64, 16
    x = rng.normal(size=n)
    rec = istft(stft(x, win, hop), win, hop, n)
    # compare on interior (windowed edges have near-zero norm)
    mid = slice(win, n - win)
    err = float(np.abs(rec[mid] - x[mid]).max())
    return {"synthetic_stft_pr": float(err < 1e-8)}
