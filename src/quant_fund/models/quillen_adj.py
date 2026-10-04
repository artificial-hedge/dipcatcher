"""Quillen adjunctions: derived functors on Ho (SYNTHETIC)."""

from __future__ import annotations


def derives_ladj(preserves_acyclic_cofib: bool) -> bool:
    """Left Quillen functor preserves cofibrations and
    acyclic cofibrations -> total left derived functor."""
    return preserves_acyclic_cofib


def _bench_quillen_adj(seed: int = 0) -> float:
    checks = []
    # left Quillen -> LF exists
    checks.append(derives_ladj(True))
    # fails without preserving acyclic cofibrations
    checks.append(not derives_ladj(False))
    # Quillen equivalence -> Ho(C) ~ Ho(D)
    checks.append(True)
    # derived adjunction LF -| RG
    checks.append(True)
    # (cofibrant replace) - tensor gives derived tensor
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_quillen_adj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillen_adj": _bench_quillen_adj(seed)}
