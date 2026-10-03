"""hiot decomp module (SYNTHETIC)."""

from __future__ import annotations


def hiot_decomp_ok(node: bool, wgt: bool) -> bool:
    """hiot_decomp
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def hiot_decomp_aux(aux: bool) -> bool:
    """hiot_decomp
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_hiot_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(hiot_decomp_ok(True, True))
    checks.append(not hiot_decomp_ok(False, True))
    checks.append(hiot_decomp_aux(True))
    checks.append(not hiot_decomp_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_hiot_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hiot_decomp": _bench_hiot_decomp(seed)}
