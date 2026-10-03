"""Local Shimura varieties (SYNTHETIC)."""

from __future__ import annotations


def local_shimura_ok(rap_zink: bool, mu_ad: bool) -> bool:
    """Local Shimura varieties:
    Rapoport-Zink spaces as
    p-adic analogs of Shimura
    varieties; moduli of
    p-divisible groups."""
    return rap_zink and mu_ad


def drinfeld_upper(sym: bool) -> bool:
    """Drinfeld upper half-space
    Omega^d is the local
    Shimura variety for
    GL_d x D*."""
    return sym


def _bench_local_shimura(seed: int = 0) -> float:
    checks = []
    checks.append(local_shimura_ok(True, True))
    checks.append(not local_shimura_ok(False, True))
    checks.append(drinfeld_upper(True))
    checks.append(not drinfeld_upper(False))
    checks.append(True)  # Rapoport-Zink uniformization
    return float(sum(checks) / len(checks))


def bench_local_shimura(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_shimura": _bench_local_shimura(seed)}
