"""Doob h-transform: conditioned BM drift (SYNTHETIC)."""

from __future__ import annotations


def h_drift(x: float, a: float, b: float) -> float:
    """BM on (a, b) conditioned to hit b: h(x) = (x - a)/(b - a),
    drift = h'/h = 1/(x - a)."""
    return 1.0 / (x - a)


def h_drift_hit_a(x: float, a: float, b: float) -> float:
    """Conditioned to hit a first: h(x) = (b - x)/(b - a), drift -1/(b-x)."""
    return -1.0 / (b - x)


def _bench_h_transform(seed: int = 0) -> float:
    checks = []
    # drift toward b is positive, toward a negative
    checks.append(h_drift(1.0, 0.0, 2.0) > 0)
    checks.append(h_drift_hit_a(1.0, 0.0, 2.0) < 0)
    # magnitudes: 1/(x-a) at x=1, a=0 is 1
    checks.append(abs(h_drift(1.0, 0.0, 2.0) - 1.0) < 1e-12)
    # drift blows up near the avoided boundary
    checks.append(h_drift(0.1, 0.0, 2.0) > h_drift(1.0, 0.0, 2.0))
    # Bessel limit a -> -inf matches 1/x drift
    checks.append(abs(h_drift(1.0, 0.0, 1e9) - 1.0) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_h_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_h_transform": _bench_h_transform(seed)}
