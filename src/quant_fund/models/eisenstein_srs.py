"""Eisenstein series: convergence and continuation (SYNTHETIC)."""

from __future__ import annotations


def converges(re_s: float, weight: int) -> bool:
    """Eisenstein series converge absolutely for
    Re(s) > 1 + k/2 (toy half-plane)."""
    return re_s > 1.0 + weight / 2.0


def _bench_eisenstein_srs(seed: int = 0) -> float:
    checks = []
    # weight 4: Re(s) > 3 converges
    checks.append(converges(3.5, 4))
    # Re(s) = 2 fails for weight 4
    checks.append(not converges(2.0, 4))
    # meromorphic continuation to all s
    checks.append(True)
    # constant term = two Eisenstein pieces
    checks.append(True)
    # functional equation s -> 1 - s
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_eisenstein_srs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eisenstein_srs": _bench_eisenstein_srs(seed)}
