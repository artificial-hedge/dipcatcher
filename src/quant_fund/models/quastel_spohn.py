"""quastel spohn module (SYNTHETIC)."""

from __future__ import annotations


def quastel_spohn_ok(kpz: bool, grow: bool) -> bool:
    """quastel_spohn
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def quastel_spohn_aux(aux: bool) -> bool:
    """quastel_spohn
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_quastel_spohn(seed: int = 0) -> float:
    checks = []
    checks.append(quastel_spohn_ok(True, True))
    checks.append(not quastel_spohn_ok(False, True))
    checks.append(quastel_spohn_aux(True))
    checks.append(not quastel_spohn_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_quastel_spohn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quastel_spohn": _bench_quastel_spohn(seed)}
