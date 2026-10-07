"""SYNTHETIC Jacobson/Karels RTT estimation.

SRTT ← (1-α)·SRTT + α·R ; RTTVAR ← (1-β)·RTTVAR + β·|SRTT−R| ;
RTO = SRTT + 4·RTTVAR. Verified: estimator tracks a step change and RTO
covers steady-state samples with high probability.
"""

from __future__ import annotations

import random


class RTTEst:
    def __init__(self, alpha: float = 0.125, beta: float = 0.25):
        self.a = alpha
        self.b = beta
        self.srtt: float | None = None
        self.var = 0.0

    def update(self, r: float) -> None:
        if self.srtt is None:
            self.srtt = r
            self.var = r / 2
            return
        self.var = (1 - self.b) * self.var + self.b * abs(self.srtt - r)
        self.srtt = (1 - self.a) * self.srtt + self.a * r

    @property
    def rto(self) -> float:
        if not (self.srtt is not None):
            raise ValueError("self.srtt is not None")
        return self.srtt + 4 * self.var


def bench_rtt_estimator(seed: int = 20261231 + 403) -> dict[str, float]:
    rng = random.Random(seed)
    track = cover = first = 0
    trials = 40
    for _ in range(trials):
        est = RTTEst()
        # phase 1: RTT ~ 50
        for _ in range(30):
            est.update(rng.gauss(50, 5))
        # phase 2: step to 150 → estimator converges
        for _ in range(60):
            est.update(rng.gauss(150, 8))
        if not (est.srtt is not None):
            raise ValueError("est.srtt is not None")
        track += int(abs(est.srtt - 150) < 15)
        # coverage: RTO above 95% of steady samples
        est2 = RTTEst()
        for _ in range(50):
            est2.update(rng.gauss(80, 6))
        hits = sum(1 for _ in range(50) if rng.gauss(80, 6) <= est2.rto)
        cover += int(hits >= 45)
        # first sample: RTO = r + 4·(r/2) = 3r
        est3 = RTTEst()
        est3.update(100.0)
        first += int(abs(est3.rto - 300.0) < 1e-9)
    return {
        "synthetic_tracks_step": float(track / trials),
        "synthetic_rto_coverage": float(cover / trials),
        "synthetic_first_rto_3r": float(first / trials),
    }
