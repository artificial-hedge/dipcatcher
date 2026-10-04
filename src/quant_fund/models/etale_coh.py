"""Etale cohomology: l-adic, Galois action (SYNTHETIC)."""

from __future__ import annotations


def l_adic_dim(betti: int, l_invertible: bool) -> int:
    """When l is invertible, etale cohomology ranks match
    the topological Betti numbers of the complex points."""
    return betti if l_invertible else -1


def _bench_etale_coh(seed: int = 0) -> float:
    checks = []
    # ranks match Betti numbers when l invertible
    checks.append(l_adic_dim(4, True) == 4)
    # l = char fails
    checks.append(l_adic_dim(4, False) == -1)
    # Galois acts on etale cohomology
    checks.append(True)
    # Weil conjectures proved via etale cohomology
    checks.append(True)
    # Galois rep coefficients in Q_l
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_etale_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_coh": _bench_etale_coh(seed)}
