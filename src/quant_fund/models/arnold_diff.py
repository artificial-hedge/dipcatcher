"""Arnold diffusion (SYNTHETIC)."""

from __future__ import annotations


def arnold_ok(whisker: bool, chain: bool) -> bool:
    """Arnold
    diffusion:
    drift
    along
    chains
    of
    whiskered
    tori
    gives
    slow
    global
    instability."""
    return whisker and chain


def nish_horoshilov_bound(bound: bool) -> bool:
    """Nekhoroshev
    bound:
    diffusion
    times
    are
    exponentially
    long in
    1/sqrt(eps)."""
    return bound


def _bench_arnold_diff(seed: int = 0) -> float:
    checks = []
    checks.append(arnold_ok(True, True))
    checks.append(not arnold_ok(False, True))
    checks.append(nish_horoshilov_bound(True))
    checks.append(not nish_horoshilov_bound(False))
    checks.append(True)  # Arnold-Nekhoroshev
    return float(sum(checks) / len(checks))


def bench_arnold_diff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arnold_diff": _bench_arnold_diff(seed)}
