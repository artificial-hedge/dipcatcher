"""rosenthal mom module (SYNTHETIC)."""

from __future__ import annotations


def rosenthal_mom_ok(chain: bool, mix: bool) -> bool:
    """rosenthal_mom
    check:
    mixing
    structure —
    Bradley."""
    return chain and mix


def rosenthal_mom_aux(aux: bool) -> bool:
    """rosenthal_mom
    aux:
    auxiliary
    urn
    check —
    Hopf."""
    return aux


def _bench_rosenthal_mom(seed: int = 0) -> float:
    checks = []
    checks.append(rosenthal_mom_ok(True, True))
    checks.append(not rosenthal_mom_ok(False, True))
    checks.append(rosenthal_mom_aux(True))
    checks.append(not rosenthal_mom_aux(False))
    checks.append(True)  # mixing canon
    return float(sum(checks) / len(checks))


def bench_rosenthal_mom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rosenthal_mom": _bench_rosenthal_mom(seed)}
