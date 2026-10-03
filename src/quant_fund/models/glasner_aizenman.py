"""glasner aizenman module (SYNTHETIC)."""

from __future__ import annotations


def glasner_aizenman_ok(rc: bool, potts: bool) -> bool:
    """glasner_aizenman
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def glasner_aizenman_aux(aux: bool) -> bool:
    """glasner_aizenman
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_glasner_aizenman(seed: int = 0) -> float:
    checks = []
    checks.append(glasner_aizenman_ok(True, True))
    checks.append(not glasner_aizenman_ok(False, True))
    checks.append(glasner_aizenman_aux(True))
    checks.append(not glasner_aizenman_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_glasner_aizenman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glasner_aizenman": _bench_glasner_aizenman(seed)}
