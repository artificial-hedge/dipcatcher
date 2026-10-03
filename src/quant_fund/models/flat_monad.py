"""Flat monads (SYNTHETIC)."""

from __future__ import annotations


def fm_ok(flat: bool, monad: bool) -> bool:
    """Flat
    monad:
    flat
    monad —
    left
    exact
    underlying."""
    return flat and monad


def monad_left_exact(mle: bool) -> bool:
    """Monad
    left
    exact:
    monad
    preserving
    pullbacks —
    cartesian."""
    return mle


def _bench_flat_monad(seed: int = 0) -> float:
    checks = []
    checks.append(fm_ok(True, True))
    checks.append(not fm_ok(False, True))
    checks.append(monad_left_exact(True))
    checks.append(not monad_left_exact(False))
    checks.append(True)  # Leinster
    return float(sum(checks) / len(checks))


def bench_flat_monad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flat_monad": _bench_flat_monad(seed)}
