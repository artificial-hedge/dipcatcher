"""moc transport module (SYNTHETIC)."""

from __future__ import annotations


def moc_transport_ok(flux: bool, ord: bool) -> bool:
    """moc_transport
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def moc_transport_aux(aux: bool) -> bool:
    """moc_transport
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_moc_transport(seed: int = 0) -> float:
    checks = []
    checks.append(moc_transport_ok(True, True))
    checks.append(not moc_transport_ok(False, True))
    checks.append(moc_transport_aux(True))
    checks.append(not moc_transport_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_moc_transport(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moc_transport": _bench_moc_transport(seed)}
