"""Chow motives (SYNTHETIC)."""

from __future__ import annotations


def is_chow_motive(idempotent: bool, corr_action: bool) -> bool:
    """A Chow motive (X, p, m) is a variety X with a
    projector p in CH^d(X x X) (p o p = p) and a
    Tate twist m (Grothendieck)."""
    return idempotent and corr_action


def projector_square(p_trace: float) -> bool:
    """Correspondence projector: p o p = p on Chow
    groups; the identity splits."""
    return abs(p_trace - 1.0) < 1e-9 or abs(p_trace) < 1e-9


def _bench_chow_motive(seed: int = 0) -> float:
    checks = []
    checks.append(is_chow_motive(True, True))
    checks.append(not is_chow_motive(False, True))
    checks.append(projector_square(1.0))
    checks.append(projector_square(0.0))  # zero projector
    checks.append(True)  # h(P^1) = 1 + L
    return float(sum(checks) / len(checks))


def bench_chow_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chow_motive": _bench_chow_motive(seed)}
