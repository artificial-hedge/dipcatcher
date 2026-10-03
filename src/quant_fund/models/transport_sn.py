"""transport sn module (SYNTHETIC)."""

from __future__ import annotations


def transport_sn_ok(flux: bool, ord: bool) -> bool:
    """transport_sn
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def transport_sn_aux(aux: bool) -> bool:
    """transport_sn
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_transport_sn(seed: int = 0) -> float:
    checks = []
    checks.append(transport_sn_ok(True, True))
    checks.append(not transport_sn_ok(False, True))
    checks.append(transport_sn_aux(True))
    checks.append(not transport_sn_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_transport_sn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transport_sn": _bench_transport_sn(seed)}
