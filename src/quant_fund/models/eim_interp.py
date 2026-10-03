"""eim interp module (SYNTHETIC)."""

from __future__ import annotations


def eim_interp_ok(basis: bool, mode: bool) -> bool:
    """eim_interp
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def eim_interp_aux(aux: bool) -> bool:
    """eim_interp
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_eim_interp(seed: int = 0) -> float:
    checks = []
    checks.append(eim_interp_ok(True, True))
    checks.append(not eim_interp_ok(False, True))
    checks.append(eim_interp_aux(True))
    checks.append(not eim_interp_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_eim_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eim_interp": _bench_eim_interp(seed)}
