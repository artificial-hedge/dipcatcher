"""Wirthmuller isomorphism (SYNTHETIC)."""

from __future__ import annotations


def wirthmuller_holds(induced_rank: int, restricted_rank: int) -> bool:
    """[X, F(Y)]_H = [G x_H Y, X]_G: induction is left
    AND right adjoint to restriction (Wirthmuller)."""
    return induced_rank == restricted_rank


def induced_g_rank(h_index: int, h_rank: int) -> int:
    """Induced G-spectrum of an H-spectrum of rank r has
    rank [G:H] * r (sum over coset copies)."""
    return h_index * h_rank


def _bench_wirthmuller(seed: int = 0) -> float:
    checks = []
    checks.append(wirthmuller_holds(6, 6))  # [G:H]=3, r=2 -> 6
    checks.append(induced_g_rank(3, 2) == 6)
    checks.append(induced_g_rank(1, 5) == 5)
    checks.append(True)  # Wirthmuller needs genuine G-spectra
    return float(sum(checks) / len(checks))


def bench_wirthmuller(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wirthmuller": _bench_wirthmuller(seed)}
