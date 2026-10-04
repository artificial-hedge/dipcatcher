"""Quillen local-devissage (SYNTHETIC)."""

from __future__ import annotations


def qld_ok(quillen: bool, ldev: bool) -> bool:
    """Quillen
    ldev:
    Quillen
    local
    devissage —
    abelian
    cat."""
    return quillen and ldev


def local_devissage(ld: bool) -> bool:
    """Local
    devissage:
    local
    devissage —
    filtration
    subcat."""
    return ld


def _bench_quillen_ldev(seed: int = 0) -> float:
    checks = []
    checks.append(qld_ok(True, True))
    checks.append(not qld_ok(False, True))
    checks.append(local_devissage(True))
    checks.append(not local_devissage(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_quillen_ldev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillen_ldev": _bench_quillen_ldev(seed)}
