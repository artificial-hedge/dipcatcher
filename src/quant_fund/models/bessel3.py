"""Bessel-3 process: BM conditioned to stay positive (SYNTHETIC)."""

from __future__ import annotations


def hit_prob_0(x: float, a: float) -> float:
    """Bessel-3: P_x(hit a < hit 0) for 0 < a < x is a/x."""
    return a / x


def drift(x: float) -> float:
    """Bessel-3 SDE dX = dW + (1/X) dt has drift 1/x."""
    return 1.0 / x


def _bench_bessel3(seed: int = 0) -> float:
    checks = []
    # probability of reaching a before 0 starting above a
    checks.append(abs(hit_prob_0(2.0, 1.0) - 0.5) < 1e-12)
    checks.append(abs(hit_prob_0(4.0, 1.0) - 0.25) < 1e-12)
    # drift is 1/x: repels from 0
    checks.append(abs(drift(2.0) - 0.5) < 1e-12)
    # Bessel never hits 0: hit_prob -> 1 as x -> inf intuition,
    # a/x -> 0 as x grows means hitting a first becomes unlikely...
    # actually a/x -> 0: from far away, unlikely to descend to a<...x
    checks.append(hit_prob_0(100.0, 1.0) < 0.02)
    # radial part of 3d BM: |B| is Bessel-3, |B|^2 is Bessel^2-3 squared
    checks.append(drift(1.0) == 1.0)
    return float(sum(checks) / len(checks))


def bench_bessel3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bessel3": _bench_bessel3(seed)}
