"""Icon categories (SYNTHETIC)."""

from __future__ import annotations


def ic_ok(icon: bool, cat: bool) -> bool:
    """Icon
    category:
    icon
    category —
    Lack
    icon."""
    return icon and cat


def icon_transf(it: bool) -> bool:
    """Icon
    transformation:
    icon
    transformation —
    lax
    icon."""
    return it


def _bench_icon_cat(seed: int = 0) -> float:
    checks = []
    checks.append(ic_ok(True, True))
    checks.append(not ic_ok(False, True))
    checks.append(icon_transf(True))
    checks.append(not icon_transf(False))
    checks.append(True)  # Lack
    return float(sum(checks) / len(checks))


def bench_icon_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_icon_cat": _bench_icon_cat(seed)}
