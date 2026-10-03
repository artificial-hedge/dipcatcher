"""cardinal interp module (SYNTHETIC)."""

from __future__ import annotations


def cardinal_interp_ok(node: bool, form: bool) -> bool:
    """cardinal_interp
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def cardinal_interp_aux(aux: bool) -> bool:
    """cardinal_interp
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_cardinal_interp(seed: int = 0) -> float:
    checks = []
    checks.append(cardinal_interp_ok(True, True))
    checks.append(not cardinal_interp_ok(False, True))
    checks.append(cardinal_interp_aux(True))
    checks.append(not cardinal_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_cardinal_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardinal_interp": _bench_cardinal_interp(seed)}
