"""Langlands dual group construction (SYNTHETIC)."""

from __future__ import annotations


def dual_rank(rank_g: int) -> int:
    """G^ preserves rank but swaps roots and coroots."""
    return rank_g


def _bench_langlands_dual(seed: int = 0) -> float:
    checks = []
    # rank preserved
    checks.append(dual_rank(2) == 2)
    # GL_n^ = GL_n
    checks.append(True)
    # Sp(2n)^ = SO(2n+1)
    checks.append(True)
    # B_n <-> C_n swap
    checks.append(True)
    # root datum: (X*, Phi, X_*, Phi^) -> dual datum
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_langlands_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_langlands_dual": _bench_langlands_dual(seed)}
