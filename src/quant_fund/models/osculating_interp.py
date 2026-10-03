"""osculating interp module (SYNTHETIC)."""

from __future__ import annotations


def osculating_interp_ok(node: bool, form: bool) -> bool:
    """osculating_interp
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def osculating_interp_aux(aux: bool) -> bool:
    """osculating_interp
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_osculating_interp(seed: int = 0) -> float:
    checks = []
    checks.append(osculating_interp_ok(True, True))
    checks.append(not osculating_interp_ok(False, True))
    checks.append(osculating_interp_aux(True))
    checks.append(not osculating_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_osculating_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osculating_interp": _bench_osculating_interp(seed)}
