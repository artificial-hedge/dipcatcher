"""Two-transformations (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(two: bool, transform: bool) -> bool:
    """Two
    transformation:
    2
    transformation —
    pseudo
    natural."""
    return two and transform


def pseudo_natural(pn: bool) -> bool:
    """Pseudo
    natural:
    pseudo
    natural
    transformation —
    coherent."""
    return pn


def _bench_two_transform(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(pseudo_natural(True))
    checks.append(not pseudo_natural(False))
    checks.append(True)  # Gray
    return float(sum(checks) / len(checks))


def bench_two_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_two_transform": _bench_two_transform(seed)}
