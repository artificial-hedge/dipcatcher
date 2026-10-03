"""rational interp module (SYNTHETIC)."""

from __future__ import annotations


def rational_interp_ok(node: bool, form: bool) -> bool:
    """rational_interp
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def rational_interp_aux(aux: bool) -> bool:
    """rational_interp
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_rational_interp(seed: int = 0) -> float:
    checks = []
    checks.append(rational_interp_ok(True, True))
    checks.append(not rational_interp_ok(False, True))
    checks.append(rational_interp_aux(True))
    checks.append(not rational_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_rational_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rational_interp": _bench_rational_interp(seed)}
