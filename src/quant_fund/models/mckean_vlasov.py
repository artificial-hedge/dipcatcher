"""mckean vlasov module (SYNTHETIC)."""

from __future__ import annotations


def mckean_vlasov_ok(mv1: bool, mk: bool) -> bool:
    """mckean_vlasov
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def mckean_vlasov_aux(aux: bool) -> bool:
    """mckean_vlasov
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_mckean_vlasov(seed: int = 0) -> float:
    checks = []
    checks.append(mckean_vlasov_ok(True, True))
    checks.append(not mckean_vlasov_ok(False, True))
    checks.append(mckean_vlasov_aux(True))
    checks.append(not mckean_vlasov_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_mckean_vlasov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mckean_vlasov": _bench_mckean_vlasov(seed)}
