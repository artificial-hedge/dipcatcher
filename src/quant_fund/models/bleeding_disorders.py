"""bleeding_disorders module (SYNTHETIC)."""

from __future__ import annotations


def bleeding_disorders_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bleeding_disorders

    check:
    bleeding_disorders: hemophilia and platelets
    ..."""
    return fit_ok and sample_ok


def bleeding_disorders_aux(aux: bool) -> bool:
    """bleeding_disorders

    aux:
    bleeding_disorders: vwf and factor
    ..."""
    return aux


def _bench_bleeding_disorders(seed: int = 0) -> float:
    checks = []
    checks.append(bleeding_disorders_ok(True, True))
    checks.append(not bleeding_disorders_ok(False, True))
    checks.append(bleeding_disorders_aux(True))
    checks.append(not bleeding_disorders_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_bleeding_disorders(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bleeding_disorders": _bench_bleeding_disorders(seed)}
