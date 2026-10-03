"""SYNTHETIC token bucket rate limiter.

Bucket holds ≤ burst tokens, refills at rate r; each request consumes 1.
Long-run throughput ≤ r (slack from initial burst amortizes), and up to
`burst` requests pass instantly at t=0.
"""

from __future__ import annotations

import random


class Bucket:
    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.t = 0.0

    def allow(self, now: float) -> bool:
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def bench_token_bucket(seed: int = 20261231 + 402) -> dict[str, float]:
    rng = random.Random(seed)
    burst_ok = rate_ok = refill_ok = 0
    trials = 40
    for _ in range(trials):
        r = rng.uniform(1, 10)
        b = rng.randrange(3, 10)
        bk = Bucket(r, b)
        # instant burst: exactly b pass at t=0
        passed0 = sum(bk.allow(0.0) for _ in range(b + 2))
        burst_ok += int(passed0 == b)
        # long-run: over horizon T, total passes ≤ b + r*T (+1 token slack)
        bk2 = Bucket(r, b)
        T = 100.0
        allowed = sum(bk2.allow(t) for t in range(int(T) + 1) if True)
        # requests offered every 0.1s
        bk3 = Bucket(r, b)
        allowed = 0
        t = 0.0
        while t <= T:
            allowed += bk3.allow(t)
            t += 0.1
        rate_ok += int(allowed <= b + r * T + 1.01)
        # refill: drain, wait 2/r seconds → ≥1 token
        bk4 = Bucket(r, b)
        for _ in range(b):
            bk4.allow(0.0)
        refill_ok += int(bk4.allow(2.0 / r) if r > 0 else False)
    return {
        "synthetic_burst_exact": float(burst_ok / trials),
        "synthetic_longrun_rate": float(rate_ok / trials),
        "synthetic_refill": float(refill_ok / trials),
    }
