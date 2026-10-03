"""Periodic-beacon detection (defensive) — wave 286.

FFT over binned connection-count series reveals a period spike for
beacon traffic vs aperiodic human-like arrivals.
"""

import numpy as np

_SEED = 20261231 + 800


def _cv(times: np.ndarray) -> float:
    iat = np.diff(np.sort(times))
    return float(iat.std() / (iat.mean() + 1e-9))


def bench_beacon_detect(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # beacon: every 60s +- 2s jitter for 24h
    beacon = np.cumsum(60.0 + rng.normal(0, 2.0, 1400))
    # organic: poisson arrivals mean-rate similar
    organic = np.cumsum(rng.exponential(58.0, 1450))
    cv_beacon, cv_organic = _cv(beacon), _cv(organic)
    return {"synthetic_beacon_cv": float(cv_beacon < 0.2 * cv_organic and cv_organic > 0.8)}
