"""Serre spectral sequence toy: E^2 = H_p(B) x H_q(F) collapses for products (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Sequence


def product_betti(b_f: Sequence[int], b_b: Sequence[int]) -> list[int]:
    """Kunneth for product B x F: b_k(E) = sum_{p+q=k} b_p(B) b_q(F)."""
    n = len(b_f) + len(b_b) - 1
    out = [0] * n
    for p, bp in enumerate(b_b):
        for q, bq in enumerate(b_f):
            out[p + q] += bp * bq
    return out


def _bench_spectral_seq_toy(seed: int = 0) -> float:
    checks = []
    s1 = [1, 1]
    # S^1 x S^1 = T^2 -> (1, 2, 1)
    checks.append(product_betti(s1, s1) == [1, 2, 1])
    # S^1 x S^2 -> (1, 1, 1, 1)
    s2 = [1, 0, 1]
    checks.append(product_betti(s1, s2) == [1, 1, 1, 1])
    # S^2 x S^2 -> (1,0,2,0,1)
    checks.append(product_betti(s2, s2) == [1, 0, 2, 0, 1])
    # torus x circle = T^3 -> (1,3,3,1)
    t2 = [1, 2, 1]
    checks.append(product_betti(t2, s1) == [1, 3, 3, 1])
    # product with point -> same
    checks.append(product_betti([1], t2) == t2)
    # Euler char multiplicative: chi(T^2 x S^2) = 0 * 2 = 0
    checks.append(sum((-1) ** i * b for i, b in enumerate(product_betti(t2, s2))) == 0)
    return float(sum(checks) / len(checks))


def bench_spectral_seq_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_seq_toy": _bench_spectral_seq_toy(seed)}
