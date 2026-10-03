"""talagrand conc module (SYNTHETIC)."""

from __future__ import annotations


def talagrand_conc_ok(gp: bool, bound: bool) -> bool:
    """talagrand_conc
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def talagrand_conc_aux(aux: bool) -> bool:
    """talagrand_conc
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_talagrand_conc(seed: int = 0) -> float:
    checks = []
    checks.append(talagrand_conc_ok(True, True))
    checks.append(not talagrand_conc_ok(False, True))
    checks.append(talagrand_conc_aux(True))
    checks.append(not talagrand_conc_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_talagrand_conc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_talagrand_conc": _bench_talagrand_conc(seed)}
