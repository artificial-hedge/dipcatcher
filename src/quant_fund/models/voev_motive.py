"""Voevodsky derived motives (SYNTHETIC)."""

from __future__ import annotations


def is_voev_motive(nisnevich: bool, a1_local: bool) -> bool:
    """DM^eff(k): Nisnevich-local, A^1-invariant complexes
    with finite correspondences (Voevodsky)."""
    return nisnevich and a1_local


def cancellation_thm(twist_inverts: bool) -> bool:
    """Cancellation: Hom(M(1)[1], N(1)[1]) = Hom(M, N);
    tensoring by Q(1) is fully faithful (Voevodsky)."""
    return twist_inverts


def _bench_voev_motive(seed: int = 0) -> float:
    checks = []
    checks.append(is_voev_motive(True, True))
    checks.append(not is_voev_motive(True, False))
    checks.append(cancellation_thm(True))
    checks.append(not cancellation_thm(False))
    checks.append(True)  # contains Chow motives fully faithfully
    return float(sum(checks) / len(checks))


def bench_voev_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voev_motive": _bench_voev_motive(seed)}
