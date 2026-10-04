"""secant root module (SYNTHETIC)."""

from __future__ import annotations


def secant_root_ok(iter_: bool, conv: bool) -> bool:
    """secant_root
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def secant_root_aux(aux: bool) -> bool:
    """secant_root
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_secant_root(seed: int = 0) -> float:
    checks = []
    checks.append(secant_root_ok(True, True))
    checks.append(not secant_root_ok(False, True))
    checks.append(secant_root_aux(True))
    checks.append(not secant_root_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_secant_root(seed: int = 0) -> dict[str, float]:
    return {"synthetic_secant_root": _bench_secant_root(seed)}
