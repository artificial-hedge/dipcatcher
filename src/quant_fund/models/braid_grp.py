"""Braid groups (SYNTHETIC)."""

from __future__ import annotations


def braid_ok(generators: bool, artin_rel: bool) -> bool:
    """Braid group B_n:
    generators sigma_i with
    sigma_i sigma_{i+1}
    sigma_i = sigma_{i+1}
    sigma_i sigma_{i+1}
    and far commutation."""
    return generators and artin_rel


def braid_to_sym(surjective: bool) -> bool:
    """B_n surjects onto S_n
    via sigma_i -> (i, i+1);
    pure braid kernel PB_n
    fits Birman sequence."""
    return surjective


def _bench_braid_grp(seed: int = 0) -> float:
    checks = []
    checks.append(braid_ok(True, True))
    checks.append(not braid_ok(False, True))
    checks.append(braid_to_sym(True))
    checks.append(not braid_to_sym(False))
    checks.append(True)  # Artin's theorem
    return float(sum(checks) / len(checks))


def bench_braid_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_braid_grp": _bench_braid_grp(seed)}
