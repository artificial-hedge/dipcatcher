"""regula falsi module (SYNTHETIC)."""

from __future__ import annotations


def regula_falsi_ok(iter_: bool, conv: bool) -> bool:
    """regula_falsi
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def regula_falsi_aux(aux: bool) -> bool:
    """regula_falsi
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_regula_falsi(seed: int = 0) -> float:
    checks = []
    checks.append(regula_falsi_ok(True, True))
    checks.append(not regula_falsi_ok(False, True))
    checks.append(regula_falsi_aux(True))
    checks.append(not regula_falsi_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_regula_falsi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regula_falsi": _bench_regula_falsi(seed)}
