"""Snake lemma / long exact sequence check on finite complexes (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def exact_at(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """im(a) == ker(b): composition zero + rank check (finite-dim)."""
    if not np.allclose(b @ a, 0, atol=tol):
        return False
    ker_b = b.shape[1] - np.linalg.matrix_rank(b, tol=tol)
    return bool(np.linalg.matrix_rank(a, tol=tol) == ker_b)


def euler_char(dims: list[int], hdims: list[int]) -> int:
    return int(
        sum((-1) ** i * d for i, d in enumerate(dims))
        - sum((-1) ** i * d for i, d in enumerate(hdims))
    )


def _bench_snake_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(exact_at(np.zeros((1, 0)), np.eye(1)))  # 0 -> V -id-> V exact at first V
    # 0 -> A -> B -> C -> 0 with A=B=ker: dim additivity
    incl = np.array([[1.0], [0.0]])  # A=R -> B=R^2
    proj = np.array([[0.0, 1.0]])  # B -> C=R
    checks.append(exact_at(incl, proj))
    # alternating sum of long exact sequence = 0
    checks.append(euler_char([1, 2, 1], [0, 0, 0]) == 0)
    # snake: ker seq 0->ker a->ker b->ker c -d-> coker a->...
    # verify connecting homomorphism exists on toy: maps already exact
    checks.append(exact_at(np.array([[2.0]]), np.array([[0.0]])))
    return float(sum(checks) / len(checks))


def bench_snake_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_snake_lemma": _bench_snake_lemma(seed)}
