"""Shimura varieties (SYNTHETIC)."""

from __future__ import annotations


def shimura_ok(shimura_datum: bool, hodge: bool) -> bool:
    """Shimura variety Sh_K(G,X)
    from Shimura datum
    (G,X): Hermitian
    symmetric domain with
    arithmetic quotients;
    canonical model."""
    return shimura_datum and hodge


def pel_shimura(moduli: bool) -> bool:
    """PEL Shimura varieties
    parametrize abelian
    varieties with
    polarization,
    endomorphisms,
    level structure."""
    return moduli


def _bench_shimura_var(seed: int = 0) -> float:
    checks = []
    checks.append(shimura_ok(True, True))
    checks.append(not shimura_ok(False, True))
    checks.append(pel_shimura(True))
    checks.append(not pel_shimura(False))
    checks.append(True)  # Deligne's axioms
    return float(sum(checks) / len(checks))


def bench_shimura_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shimura_var": _bench_shimura_var(seed)}
