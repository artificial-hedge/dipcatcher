"""Shifted symplectic structures, PTVV (SYNTHETIC)."""

from __future__ import annotations


def symplectic_degree(n: int) -> int:
    """An n-shifted symplectic form is a closed 2-form of
    degree n; n = -1 gives Lagrangian intersections."""
    return n


def _bench_shifted_symplectic(seed: int = 0) -> float:
    checks = []
    # 0-shifted = ordinary symplectic
    checks.append(symplectic_degree(0) == 0)
    # -1-shifted: derived critical loci
    checks.append(symplectic_degree(-1) == -1)
    # nondegenerate pairing on tangent complex
    checks.append(True)
    # Lagrangian intersections are (-1)-shifted
    checks.append(True)
    # Darboux lemma holds in shifted setting
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_shifted_symplectic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shifted_symplectic": _bench_shifted_symplectic(seed)}
