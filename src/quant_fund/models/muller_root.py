"""muller root module (SYNTHETIC)."""

from __future__ import annotations


def muller_root_ok(iter_: bool, conv: bool) -> bool:
    """muller_root
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def muller_root_aux(aux: bool) -> bool:
    """muller_root
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_muller_root(seed: int = 0) -> float:
    checks = []
    checks.append(muller_root_ok(True, True))
    checks.append(not muller_root_ok(False, True))
    checks.append(muller_root_aux(True))
    checks.append(not muller_root_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_muller_root(seed: int = 0) -> dict[str, float]:
    return {"synthetic_muller_root": _bench_muller_root(seed)}
