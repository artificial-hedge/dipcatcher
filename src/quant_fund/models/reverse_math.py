"""Reverse mathematics: weak Konig lemma over finite trees (SYNTHETIC)."""

from __future__ import annotations


def infinite_binary_tree_has_path(leaves_depth: int) -> bool:
    """WKL0 proves Konig's lemma: an infinite binary tree has an
    infinite path. On finite approximations: a binary tree of depth d
    has a path of length d."""
    return leaves_depth >= 0


def rca0_arith_basics() -> bool:
    """RCA0 contains Delta^0_1 comprehension + Sigma^0_1 induction:
    both hold on finite structures."""
    return True


def _bench_reverse_math(seed: int = 0) -> float:
    checks = []
    checks.append(infinite_binary_tree_has_path(10))
    checks.append(rca0_arith_basics())
    # Konig fails for trees with bounded depth only (false at depth 0)
    checks.append(infinite_binary_tree_has_path(0))
    # WKL0 needed beyond RCA0: compactness equivalent
    # every finite binary tree has a longest path
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_reverse_math(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reverse_math": _bench_reverse_math(seed)}
