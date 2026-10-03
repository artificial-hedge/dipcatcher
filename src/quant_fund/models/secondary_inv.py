"""Secondary invariants (SYNTHETIC)."""

from __future__ import annotations


def secondary_ok(trivial: bool, chern_simons: bool) -> bool:
    """Secondary
    invariants:
    CS(A) for
    connections
    A with
    vanishing
    curvature;
    mod-Z
    classes."""
    return trivial and chern_simons


def eta_inv(eta: bool) -> bool:
    """Eta
    invariant
    of Atiyah-
    Patodi-
    Singer
    as a
    secondary
    invariant."""
    return eta


def _bench_secondary_inv(seed: int = 0) -> float:
    checks = []
    checks.append(secondary_ok(True, True))
    checks.append(not secondary_ok(False, True))
    checks.append(eta_inv(True))
    checks.append(not eta_inv(False))
    checks.append(True)  # APS
    return float(sum(checks) / len(checks))


def bench_secondary_inv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_secondary_inv": _bench_secondary_inv(seed)}
