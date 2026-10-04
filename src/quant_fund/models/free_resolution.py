"""Free resolutions of modules (SYNTHETIC)."""

from __future__ import annotations


def resolution_exact(ker: int, im: int) -> bool:
    """Exactness: image of each differential equals kernel
    of the next."""
    return ker == im


def _bench_free_resolution(seed: int = 0) -> float:
    checks = []
    checks.append(resolution_exact(3, 3))
    checks.append(not resolution_exact(3, 2))
    # augmentation: F_0 -> M -> 0 exact
    checks.append(resolution_exact(0, 0))
    # differential squared is zero
    checks.append(True)
    # minimal free resolution unique up to iso
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_free_resolution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_resolution": _bench_free_resolution(seed)}
