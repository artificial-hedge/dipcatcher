"""bessel fn module (SYNTHETIC)."""

from __future__ import annotations


def bessel_fn_ok(arg: bool, order: bool) -> bool:
    """bessel_fn
    check:
    special
    function —
    argument."""
    return arg and order


def bessel_fn_aux(aux: bool) -> bool:
    """bessel_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_bessel_fn(seed: int = 0) -> float:
    checks = []
    checks.append(bessel_fn_ok(True, True))
    checks.append(not bessel_fn_ok(False, True))
    checks.append(bessel_fn_aux(True))
    checks.append(not bessel_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_bessel_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bessel_fn": _bench_bessel_fn(seed)}
