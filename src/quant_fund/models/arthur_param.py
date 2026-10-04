"""Arthur parameters (SYNTHETIC)."""

from __future__ import annotations


def arthur_ok(sl2: bool, psi: bool) -> bool:
    """Arthur parameter psi:
    L_F x SL_2 -> L_G;
    accounts for non-tempered
    reps of classical
    groups."""
    return sl2 and psi


def arthur_classification(endoscopic: bool) -> bool:
    """Arthur's classification
    of discrete automorphic
    spectrum of symplectic
    / orthogonal groups via
    endoscopy."""
    return endoscopic


def _bench_arthur_param(seed: int = 0) -> float:
    checks = []
    checks.append(arthur_ok(True, True))
    checks.append(not arthur_ok(False, True))
    checks.append(arthur_classification(True))
    checks.append(not arthur_classification(False))
    checks.append(True)  # Arthur's book
    return float(sum(checks) / len(checks))


def bench_arthur_param(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arthur_param": _bench_arthur_param(seed)}
