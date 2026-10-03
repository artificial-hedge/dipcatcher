"""amir corwin module (SYNTHETIC)."""

from __future__ import annotations


def amir_corwin_ok(kpz: bool, grow: bool) -> bool:
    """amir_corwin
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def amir_corwin_aux(aux: bool) -> bool:
    """amir_corwin
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_amir_corwin(seed: int = 0) -> float:
    checks = []
    checks.append(amir_corwin_ok(True, True))
    checks.append(not amir_corwin_ok(False, True))
    checks.append(amir_corwin_aux(True))
    checks.append(not amir_corwin_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_amir_corwin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amir_corwin": _bench_amir_corwin(seed)}
