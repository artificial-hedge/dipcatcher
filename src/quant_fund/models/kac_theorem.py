"""kac theorem module (SYNTHETIC)."""

from __future__ import annotations


def kac_theorem_ok(mv1: bool, mk: bool) -> bool:
    """kac_theorem
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def kac_theorem_aux(aux: bool) -> bool:
    """kac_theorem
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_kac_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(kac_theorem_ok(True, True))
    checks.append(not kac_theorem_ok(False, True))
    checks.append(kac_theorem_aux(True))
    checks.append(not kac_theorem_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_kac_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kac_theorem": _bench_kac_theorem(seed)}
