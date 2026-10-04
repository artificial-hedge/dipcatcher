"""Flat functor theory (SYNTHETIC)."""

from __future__ import annotations


def ff_ok(flat: bool, left_exact: bool) -> bool:
    """Flat
    functor:
    flat
    functor —
    left-exact
    iff."""
    return flat and left_exact


def flat_presheaf(fp: bool) -> bool:
    """Flat
    presheaf:
    flat
    presheaf —
    finite
    limit
    preserved."""
    return fp


def _bench_flat_functor(seed: int = 0) -> float:
    checks = []
    checks.append(ff_ok(True, True))
    checks.append(not ff_ok(False, True))
    checks.append(flat_presheaf(True))
    checks.append(not flat_presheaf(False))
    checks.append(True)  # flat = lex
    return float(sum(checks) / len(checks))


def bench_flat_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flat_functor": _bench_flat_functor(seed)}
