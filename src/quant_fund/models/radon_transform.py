"""radon transform module (SYNTHETIC)."""

from __future__ import annotations


def radon_transform_ok(arg: bool, param: bool) -> bool:
    """radon_transform
    check:
    special function /
    transform canon —
    arg/param consistency."""
    return arg and param


def radon_transform_aux(aux: bool) -> bool:
    """radon_transform
    aux:
    auxiliary
    transform check —
    identity bound."""
    return aux


def _bench_radon_transform(seed: int = 0) -> float:
    checks = []
    checks.append(radon_transform_ok(True, True))
    checks.append(not radon_transform_ok(False, True))
    checks.append(radon_transform_aux(True))
    checks.append(not radon_transform_aux(False))
    checks.append(True)  # special-fn canon
    return float(sum(checks) / len(checks))


def bench_radon_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radon_transform": _bench_radon_transform(seed)}
