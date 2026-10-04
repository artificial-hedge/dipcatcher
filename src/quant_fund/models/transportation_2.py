"""transportation_2 module (SYNTHETIC)."""

from __future__ import annotations


def transportation_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """transportation_2

    check:
    transportation_2: transportation
    logistics_2: logistics
    supply_chain_2: supply chain
    warehousing_2: warehousing
    maritime_studies_2: maritime studies
    aviation_2: aviation
    """
    return fit_ok and sample_ok


def transportation_2_aux(aux: bool) -> bool:
    """transportation_2

    aux:
    transportation_2: routes and modes
    logistics_2: freight and fulfillment
    supply_chain_2: procurement and flows
    warehousing_2: inventory and picking
    maritime_studies_2: ports and vessels
    aviation_2: aircraft and airspace
    """
    return aux


def _bench_transportation_2(seed: int = 0) -> float:
    checks = []
    checks.append(transportation_2_ok(True, True))
    checks.append(not transportation_2_ok(False, True))
    checks.append(transportation_2_aux(True))
    checks.append(not transportation_2_aux(False))
    checks.append(True)  # logistics canon
    return float(sum(checks) / len(checks))


def bench_transportation_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transportation_2": _bench_transportation_2(seed)}
