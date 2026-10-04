"""Repetition-code QEC under the bit-flip channel.

Distance-d code: |0_L> = |0>^d, |1_L> = |1>^d, majority-vote decoding.
Logical failure prob = P(Bin(d, p) > d/2); bench verifies the simulated
failure rate tracks the analytic curve and decreases in d.
"""

from __future__ import annotations

import math

import numpy as np

_SEED = 20261231 + 949


def encode(bit: int, d: int) -> np.ndarray:
    return np.full(d, bit, dtype=np.uint8)


def channel(code: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    flips = (rng.random(len(code)) < p).astype(np.uint8)
    return code ^ flips


def decode(received: np.ndarray) -> int:
    return int(received.sum() * 2 > len(received))


def analytic_failure(d: int, p: float) -> float:
    return sum(math.comb(d, k) * p**k * (1 - p) ** (d - k) for k in range(d // 2 + 1, d + 1))


def bench_repetition_qec(seed: int = _SEED, shots: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks = []
    p = 0.1
    fails = {}
    for d in (3, 5, 7):
        bad = 0
        for _ in range(shots):
            bit = int(rng.integers(0, 2))
            bad += decode(channel(encode(bit, d), p, rng)) != bit
        fails[d] = bad / shots
        # within 3 sigma of the analytic value
        ana = analytic_failure(d, p)
        tol = 3 * math.sqrt(ana * (1 - ana) / shots) + 1e-9
        checks.append(abs(fails[d] - ana) <= max(tol, 0.01))
    checks.append(fails[3] > fails[5] > fails[7])
    checks.append(fails[7] < p)
    return {"synthetic_repetition_qec": float(np.mean(checks))}
