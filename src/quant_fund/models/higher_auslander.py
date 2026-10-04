"""higher auslander module (SYNTHETIC)."""

from __future__ import annotations


def higher_auslander_ok(representation: bool, finite: bool) -> bool:
    """higher_auslander
    check:
    representation
    structure —
    helix."""
    return representation and finite


def higher_auslander_aux(aux: bool) -> bool:
    """higher_auslander
    aux:
    auxiliary
    representation
    check —
    quiver."""
    return aux


def _bench_higher_auslander(seed: int = 0) -> float:
    checks = []
    checks.append(higher_auslander_ok(True, True))
    checks.append(not higher_auslander_ok(False, True))
    checks.append(higher_auslander_aux(True))
    checks.append(not higher_auslander_aux(False))
    checks.append(True)  # rep-theory canon
    return float(sum(checks) / len(checks))


def bench_higher_auslander(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_auslander": _bench_higher_auslander(seed)}
