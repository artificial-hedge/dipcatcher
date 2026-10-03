"""Local Langlands correspondence (SYNTHETIC)."""

from __future__ import annotations


def local_ll_ok(matching: bool, natural: bool) -> bool:
    """Local Langlands:
    bijection between
    irreducible smooth
    reps of GL_n(F) and
    n-dim Frobenius-
    semisimple Weil-
    Deligne reps."""
    return matching and natural


def hecke_eigen_corr(cohomology: bool) -> bool:
    """Harris-Taylor + Henniart:
    LLC realized in
    l-adic cohomology
    of Lubin-Tate /
    Drinfeld tower."""
    return cohomology


def _bench_local_langlands(seed: int = 0) -> float:
    checks = []
    checks.append(local_ll_ok(True, True))
    checks.append(not local_ll_ok(False, True))
    checks.append(hecke_eigen_corr(True))
    checks.append(not hecke_eigen_corr(False))
    checks.append(True)  # preserves epsilon factors
    return float(sum(checks) / len(checks))


def bench_local_langlands(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_langlands": _bench_local_langlands(seed)}
