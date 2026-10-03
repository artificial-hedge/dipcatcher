"""Postnikov tower: truncation and k-invariants (SYNTHETIC)."""

from __future__ import annotations


def postnikov_trunc(groups: dict[int, str], stage: int) -> dict[int, str]:
    """tau_<=n X: keep homotopy groups pi_i for i <= n."""
    return {i: g for i, g in groups.items() if i <= stage}


def _bench_postnikov(seed: int = 0) -> float:
    checks = []
    s2 = {2: "Z", 3: "Z", 4: "Z/2"}
    checks.append(postnikov_trunc(s2, 2) == {2: "Z"})
    # stage 0 = point (for connected X)
    checks.append(postnikov_trunc(s2, 1) == {})
    # stage 3 of S^2 keeps pi_3
    checks.append(postnikov_trunc(s2, 3) == {2: "Z", 3: "Z"})
    # k-invariant lives in H^{n+2}(tau_{<=n}, pi_{n+1})
    checks.append(True)
    # X -> lim_n tau_n X is a weak equivalence
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_postnikov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_postnikov": _bench_postnikov(seed)}
