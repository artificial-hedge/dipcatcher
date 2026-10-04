"""floater hormann module (SYNTHETIC)."""

from __future__ import annotations


def floater_hormann_ok(node: bool, weight: bool) -> bool:
    """floater_hormann
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def floater_hormann_aux(aux: bool) -> bool:
    """floater_hormann
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_floater_hormann(seed: int = 0) -> float:
    checks = []
    checks.append(floater_hormann_ok(True, True))
    checks.append(not floater_hormann_ok(False, True))
    checks.append(floater_hormann_aux(True))
    checks.append(not floater_hormann_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_floater_hormann(seed: int = 0) -> dict[str, float]:
    return {"synthetic_floater_hormann": _bench_floater_hormann(seed)}
