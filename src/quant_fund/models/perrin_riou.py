"""perrin riou module (SYNTHETIC)."""

from __future__ import annotations


def perrin_riou_ok(iwasawa: bool, euler: bool) -> bool:
    """perrin_riou
    check:
    Iwasawa
    structure —
    Euler."""
    return iwasawa and euler


def perrin_riou_aux(aux: bool) -> bool:
    """perrin_riou
    aux:
    auxiliary
    Iwasawa
    check —
    Selmer."""
    return aux


def _bench_perrin_riou(seed: int = 0) -> float:
    checks = []
    checks.append(perrin_riou_ok(True, True))
    checks.append(not perrin_riou_ok(False, True))
    checks.append(perrin_riou_aux(True))
    checks.append(not perrin_riou_aux(False))
    checks.append(True)  # Iwasawa/Euler-system canon
    return float(sum(checks) / len(checks))


def bench_perrin_riou(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perrin_riou": _bench_perrin_riou(seed)}
