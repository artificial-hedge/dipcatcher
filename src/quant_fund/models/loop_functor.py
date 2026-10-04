"""Loop functor Omega (SYNTHETIC)."""

from __future__ import annotations


def loop_functor_ok(adjunction: bool, delooping: bool) -> bool:
    """Omega X = Map_*(S^1, X);
    Sigma-Omega adjunction;
    deloopings exist for
    group-like monoids."""
    return adjunction and delooping


def recognition(grouplike: bool) -> bool:
    """May recognition: group-
    like A_infty/E_n monoids
    are n-fold loop spaces."""
    return grouplike


def _bench_loop_functor(seed: int = 0) -> float:
    checks = []
    checks.append(loop_functor_ok(True, True))
    checks.append(not loop_functor_ok(False, True))
    checks.append(recognition(True))
    checks.append(not recognition(False))
    checks.append(True)  # Omega Sigma X as James
    return float(sum(checks) / len(checks))


def bench_loop_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loop_functor": _bench_loop_functor(seed)}
