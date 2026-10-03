"""Egress-volume exfil detection (defensive) — wave 286.

Robust z-score (median + MAD) on daily egress flags volume spikes while
ignoring heavy-tailed baseline.
"""

import numpy as np

_SEED = 20261231 + 804


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    out: np.ndarray = 0.6745 * (x - med) / mad
    return out


def bench_exfil_zscore(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    base = rng.lognormal(10, 0.4, 59)  # ~22-30GB baseline days
    eg = np.concatenate([base, [base.max() * 8.0]])
    z = _robust_z(eg)
    return {"synthetic_exfil_z": float(z[-1] > 6.0 and z[:-1].max() < 6.0)}
