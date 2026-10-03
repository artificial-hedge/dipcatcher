"""Pseudo-naturality (SYNTHETIC)."""

from __future__ import annotations


def pn_ok(pseudo: bool, natural: bool) -> bool:
    """Pseudo:
    pseudo-
    natural
    transformation —
    Gray
    pseudo."""
    return pseudo and natural


def pseudonat_square(ps: bool) -> bool:
    """Pseudo
    naturality:
    pseudo-naturality
    2-
    cell
    square —
    Gray
    square."""
    return ps


def _bench_pseudo_naturality(seed: int = 0) -> float:
    checks = []
    checks.append(pn_ok(True, True))
    checks.append(not pn_ok(False, True))
    checks.append(pseudonat_square(True))
    checks.append(not pseudonat_square(False))
    checks.append(True)  # Gray
    return float(sum(checks) / len(checks))


def bench_pseudo_naturality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudo_naturality": _bench_pseudo_naturality(seed)}
