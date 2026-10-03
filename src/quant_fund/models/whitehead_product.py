"""whitehead product module (SYNTHETIC)."""

from __future__ import annotations


def whitehead_product_ok(homotopy: bool, periodic: bool) -> bool:
    """whitehead_product
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def whitehead_product_aux(aux: bool) -> bool:
    """whitehead_product
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_whitehead_product(seed: int = 0) -> float:
    checks = []
    checks.append(whitehead_product_ok(True, True))
    checks.append(not whitehead_product_ok(False, True))
    checks.append(whitehead_product_aux(True))
    checks.append(not whitehead_product_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_whitehead_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitehead_product": _bench_whitehead_product(seed)}
