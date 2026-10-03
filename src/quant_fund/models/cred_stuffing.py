"""Credential-stuffing detection (defensive) — wave 286.

Attack signature: sustained high failed-login rate with high unique-IP
ratio; organic failures are low-rate and repeat-IP.
"""

import numpy as np

_SEED = 20261231 + 802


def stuffing_score(fails: np.ndarray, unique_ips: np.ndarray, totals: np.ndarray) -> float:
    fail_rate = fails / np.maximum(totals, 1)
    ip_ratio = unique_ips / np.maximum(fails, 1)
    return float(np.mean(fail_rate * ip_ratio))


def bench_cred_stuffing(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # 24 hourly buckets; hours 8..11 = attack
    totals = rng.poisson(1000, 24).astype(float)
    fails = rng.poisson(15, 24).astype(float)
    ips = np.minimum(fails * rng.uniform(0.3, 0.6, 24), fails).astype(int)
    fails[8:12] += rng.poisson(300, 4)
    ips[8:12] = (fails[8:12] * rng.uniform(0.85, 0.98, 4)).astype(int)
    atk = stuffing_score(fails[8:12], ips[8:12], totals[8:12])
    benign_idx = [i for i in range(24) if not 8 <= i < 12]
    ben = stuffing_score(fails[benign_idx], ips[benign_idx], totals[benign_idx])
    return {"synthetic_cred_stuffing": float(atk > 5 * ben)}
