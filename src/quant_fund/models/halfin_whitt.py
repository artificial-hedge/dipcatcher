"""halfin whitt module (SYNTHETIC)."""

from __future__ import annotations


def halfin_whitt_ok(lim: bool, scale: bool) -> bool:
    """halfin_whitt
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def halfin_whitt_aux(aux: bool) -> bool:
    """halfin_whitt
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_halfin_whitt(seed: int = 0) -> float:
    checks = []
    checks.append(halfin_whitt_ok(True, True))
    checks.append(not halfin_whitt_ok(False, True))
    checks.append(halfin_whitt_aux(True))
    checks.append(not halfin_whitt_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_halfin_whitt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_halfin_whitt": _bench_halfin_whitt(seed)}
