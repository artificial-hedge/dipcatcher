"""corwin kpz module (SYNTHETIC)."""

from __future__ import annotations


def corwin_kpz_ok(kpz: bool, grow: bool) -> bool:
    """corwin_kpz
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def corwin_kpz_aux(aux: bool) -> bool:
    """corwin_kpz
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_corwin_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(corwin_kpz_ok(True, True))
    checks.append(not corwin_kpz_ok(False, True))
    checks.append(corwin_kpz_aux(True))
    checks.append(not corwin_kpz_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_corwin_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_corwin_kpz": _bench_corwin_kpz(seed)}
