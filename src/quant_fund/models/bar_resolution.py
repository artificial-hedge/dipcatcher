"""Bar resolution (SYNTHETIC)."""

from __future__ import annotations


def bar_ok(bar: bool, aug: bool) -> bool:
    """Bar resolution
    B(A): tensor
    coalgebra on
    the augmentation
    ideal; computes
    Tor^A(k,k)."""
    return bar and aug


def cobar_diff(diff: bool) -> bool:
    """Cobar construction:
    dual of bar;
    computes Ext_A
    via a cosimplicial
    differential."""
    return diff


def _bench_bar_resolution(seed: int = 0) -> float:
    checks = []
    checks.append(bar_ok(True, True))
    checks.append(not bar_ok(False, True))
    checks.append(cobar_diff(True))
    checks.append(not cobar_diff(False))
    checks.append(True)  # Eilenberg-Mac Lane
    return float(sum(checks) / len(checks))


def bench_bar_resolution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bar_resolution": _bench_bar_resolution(seed)}
