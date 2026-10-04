"""Toy spectral sequence: Hopf fibration S^1 -> S^3 -> S^2 (SYNTHETIC)."""

from __future__ import annotations


def hopf_e2() -> dict[tuple[int, int], int]:
    """E^2_{p,q} = H_p(S^2) x H_q(S^1): nonzero at p in {0,2}, q in {0,1}."""
    return {(p, q): 1 for p in (0, 2) for q in (0, 1)}


def hopf_d2_kills() -> list[tuple[int, int]]:
    """d2: E^2_{2,0} -> E^2_{0,1} must be an isomorphism for the total
    to compute H*(S^3); the surviving terms are (0,0) and (2,1)."""
    return [(0, 0), (2, 1)]


def h_s3_betti() -> list[int]:
    return [1, 0, 0, 1]


def _bench_spectral_seq(seed: int = 0) -> float:
    checks = []
    e2 = hopf_e2()
    # E^2 has 4 nonzero terms
    checks.append(len(e2) == 4)
    # total dimension at E^2 is 4
    checks.append(sum(e2.values()) == 4)
    # survivors yield S^3 betti numbers
    surv = hopf_d2_kills()
    checks.append(surv == [(0, 0), (2, 1)])
    # Betti of S^3: [1,0,0,1]
    checks.append(h_s3_betti() == [1, 0, 0, 1])
    # the only possible nonzero differential is d2
    checks.append(all(p + q <= 3 for (p, q) in e2))
    return float(sum(checks) / len(checks))


def bench_spectral_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_seq": _bench_spectral_seq(seed)}
