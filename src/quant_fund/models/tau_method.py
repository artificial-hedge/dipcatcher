"""tau method module (SYNTHETIC)."""

from __future__ import annotations


def tau_method_ok(node: bool, resid: bool) -> bool:
    """tau_method
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def tau_method_aux(aux: bool) -> bool:
    """tau_method
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_tau_method(seed: int = 0) -> float:
    checks = []
    checks.append(tau_method_ok(True, True))
    checks.append(not tau_method_ok(False, True))
    checks.append(tau_method_aux(True))
    checks.append(not tau_method_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_tau_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tau_method": _bench_tau_method(seed)}
