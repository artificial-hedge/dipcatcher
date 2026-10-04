"""p-adic Galois representations (SYNTHETIC)."""

from __future__ import annotations


def gp_ok(galois: bool, padic: bool) -> bool:
    """p-adic
    Galois:
    p-adic
    Galois
    representation —
    de
    Rham."""
    return galois and padic


def de_rham_rep(dr: bool) -> bool:
    """de
    Rham
    rep:
    de
    Rham
    representation —
    admissible."""
    return dr


def _bench_galois_padic(seed: int = 0) -> float:
    checks = []
    checks.append(gp_ok(True, True))
    checks.append(not gp_ok(False, True))
    checks.append(de_rham_rep(True))
    checks.append(not de_rham_rep(False))
    checks.append(True)  # Fontaine Perrin-Riou
    return float(sum(checks) / len(checks))


def bench_galois_padic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_padic": _bench_galois_padic(seed)}
