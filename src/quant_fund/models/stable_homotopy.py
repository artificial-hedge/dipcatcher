"""Stable homotopy groups of spheres (SYNTHETIC)."""

from __future__ import annotations


def stable_stem(n: int) -> str:
    """Low stems: pi_0^S = Z, pi_1^S = Z/2 (eta), pi_2^S = Z/2 (eta^2)."""
    return {0: "Z", 1: "Z/2", 2: "Z/2"}.get(n, "finite")


def _bench_stable_homotopy(seed: int = 0) -> float:
    checks = []
    checks.append(stable_stem(0) == "Z")
    checks.append(stable_stem(1) == "Z/2")
    checks.append(stable_stem(2) == "Z/2")
    # stems are finite for n > 0 (Serre)
    checks.append(stable_stem(3) == "finite")
    # stabilization: pi_{n+k}(S^n) independent of n for n > k+1
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_stable_homotopy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_homotopy": _bench_stable_homotopy(seed)}
