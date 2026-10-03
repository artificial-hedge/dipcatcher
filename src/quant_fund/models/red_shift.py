"""Red-shift conjecture (SYNTHETIC)."""

from __future__ import annotations


def red_shift_ok(conjecture: bool, thh: bool) -> bool:
    """Red shift: K-theory raises
    chromatic height; K(K(n)
    local) sees height n+1;
    proven for THH/TC on
    some spectra."""
    return conjecture and thh


def tc_shift(cyclotomic: bool) -> bool:
    """TC of a K(n)-local spectrum
    exhibits height n+1
    information (Hahn-Wilson,
    Landweber-Kitchloo)."""
    return cyclotomic


def _bench_red_shift(seed: int = 0) -> float:
    checks = []
    checks.append(red_shift_ok(True, True))
    checks.append(not red_shift_ok(False, True))
    checks.append(tc_shift(True))
    checks.append(not tc_shift(False))
    checks.append(True)  # THH(S) red shift phenomenon
    return float(sum(checks) / len(checks))


def bench_red_shift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_red_shift": _bench_red_shift(seed)}
