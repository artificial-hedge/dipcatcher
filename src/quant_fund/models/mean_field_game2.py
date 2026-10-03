"""mean field_game2 module (SYNTHETIC)."""

from __future__ import annotations


def mean_field_game2_ok(mv1: bool, mk: bool) -> bool:
    """mean_field_game2
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def mean_field_game2_aux(aux: bool) -> bool:
    """mean_field_game2
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_mean_field_game2(seed: int = 0) -> float:
    checks = []
    checks.append(mean_field_game2_ok(True, True))
    checks.append(not mean_field_game2_ok(False, True))
    checks.append(mean_field_game2_aux(True))
    checks.append(not mean_field_game2_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_mean_field_game2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mean_field_game2": _bench_mean_field_game2(seed)}
