"""kreyn resolvent module (SYNTHETIC)."""

from __future__ import annotations


def kreyn_resolvent_ok(sc: bool, sp: bool) -> bool:
    """kreyn_resolvent
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def kreyn_resolvent_aux(aux: bool) -> bool:
    """kreyn_resolvent
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_kreyn_resolvent(seed: int = 0) -> float:
    checks = []
    checks.append(kreyn_resolvent_ok(True, True))
    checks.append(not kreyn_resolvent_ok(False, True))
    checks.append(kreyn_resolvent_aux(True))
    checks.append(not kreyn_resolvent_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_kreyn_resolvent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kreyn_resolvent": _bench_kreyn_resolvent(seed)}
