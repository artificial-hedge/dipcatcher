"""Weighted overlap-add synthesis: hop consistency check."""

import numpy as np

_SEED = 20261231 + 750


def wola(frames: np.ndarray, hop: int) -> np.ndarray:
    win = frames.shape[1]
    n = (len(frames) - 1) * hop + win
    out = np.zeros(n)
    w = np.hanning(win)
    for i, f in enumerate(frames):
        out[i * hop : i * hop + win] += f * w
    return out


def bench_wola_synth(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    win, hop, n = 64, 16, 256
    x = rng.normal(size=n)
    w = np.hanning(win)
    frames = np.array([x[s : s + win] * w for s in range(0, n - win + 1, hop)])
    rec = wola(frames, hop)
    # interior where window energy is full
    norm = np.zeros(n)
    for i in range(len(frames)):
        norm[i * hop : i * hop + win] += w**2
    mid = slice(win, n - win)
    err = float(np.abs(rec[mid] / norm[mid] - x[mid]).max())
    return {"synthetic_wola_exact": float(err < 1e-9)}
