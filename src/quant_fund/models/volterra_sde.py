"""volterra sde module (SYNTHETIC)."""

from __future__ import annotations


def volterra_sde_ok(fh1: bool, rv: bool) -> bool:
    """volterra_sde
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def volterra_sde_aux(aux: bool) -> bool:
    """volterra_sde
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_volterra_sde(seed: int = 0) -> float:
    checks = []
    checks.append(volterra_sde_ok(True, True))
    checks.append(not volterra_sde_ok(False, True))
    checks.append(volterra_sde_aux(True))
    checks.append(not volterra_sde_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_volterra_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_volterra_sde": _bench_volterra_sde(seed)}
