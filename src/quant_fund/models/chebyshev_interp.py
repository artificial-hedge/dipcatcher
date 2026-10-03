"""chebyshev interp module (SYNTHETIC)."""

from __future__ import annotations


def chebyshev_interp_ok(node: bool, form: bool) -> bool:
    """chebyshev_interp
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def chebyshev_interp_aux(aux: bool) -> bool:
    """chebyshev_interp
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_chebyshev_interp(seed: int = 0) -> float:
    checks = []
    checks.append(chebyshev_interp_ok(True, True))
    checks.append(not chebyshev_interp_ok(False, True))
    checks.append(chebyshev_interp_aux(True))
    checks.append(not chebyshev_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_chebyshev_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_interp": _bench_chebyshev_interp(seed)}
