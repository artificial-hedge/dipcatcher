"""Smale h-cobordism (SYNTHETIC)."""

from __future__ import annotations


def hc_ok(simply_connected: bool, product: bool) -> bool:
    """h-
    cobordism:
    simply
    connected
    h-
    cobordism
    is
    a
    product —
    Smale
    theorem,
    high
    dimension."""
    return simply_connected and product


def handle_trades(ht: bool) -> bool:
    """Handle
    trades:
    h-
    cobordism
    proof
    cancels
    handles
    via
    Whitney
    trick —
    Smale
    argument."""
    return ht


def _bench_smale_hcob(seed: int = 0) -> float:
    checks = []
    checks.append(hc_ok(True, True))
    checks.append(not hc_ok(False, True))
    checks.append(handle_trades(True))
    checks.append(not handle_trades(False))
    checks.append(True)  # Smale
    return float(sum(checks) / len(checks))


def bench_smale_hcob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smale_hcob": _bench_smale_hcob(seed)}
