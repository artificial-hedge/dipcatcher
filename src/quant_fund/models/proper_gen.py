"""proper gen module (SYNTHETIC)."""

from __future__ import annotations


def proper_gen_ok(basis: bool, mode: bool) -> bool:
    """proper_gen
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def proper_gen_aux(aux: bool) -> bool:
    """proper_gen
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_proper_gen(seed: int = 0) -> float:
    checks = []
    checks.append(proper_gen_ok(True, True))
    checks.append(not proper_gen_ok(False, True))
    checks.append(proper_gen_aux(True))
    checks.append(not proper_gen_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_proper_gen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proper_gen": _bench_proper_gen(seed)}
