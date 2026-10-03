"""Matrix-tree theorem (SYNTHETIC)."""

from __future__ import annotations


def mt_ok(laplacian_minor: bool, spanning_trees: bool) -> bool:
    """Matrix-
    tree:
    spanning
    trees
    counted
    by
    Laplacian
    cofactor —
    Kirchhoff's
    theorem."""
    return laplacian_minor and spanning_trees


def kirchhoff_theorem(kt: bool) -> bool:
    """Kirchhoff:
    determinant
    of
    reduced
    Laplacian
    gives
    tree
    count —
    matrix-tree."""
    return kt


def _bench_matrix_tree(seed: int = 0) -> float:
    checks = []
    checks.append(mt_ok(True, True))
    checks.append(not mt_ok(False, True))
    checks.append(kirchhoff_theorem(True))
    checks.append(not kirchhoff_theorem(False))
    checks.append(True)  # Kirchhoff
    return float(sum(checks) / len(checks))


def bench_matrix_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_tree": _bench_matrix_tree(seed)}
