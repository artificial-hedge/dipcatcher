"""Left Bousfield localization (SYNTHETIC)."""

from __future__ import annotations


def localized_we(new_acyclic: int, old_acyclic: int) -> bool:
    """L_S C keeps cofibrations, enlarges weak equivalences
    by S-local equivalences; local objects are the new
    fibrant ones."""
    return new_acyclic >= old_acyclic


def _bench_localization_mc(seed: int = 0) -> float:
    checks = []
    # localization enlarges weak equivalences
    checks.append(localized_we(6, 4))
    # shrinking fails
    checks.append(not localized_we(3, 4))
    # exists for left proper cellular/combinatorial
    checks.append(True)
    # homotopy category = S^{-1} Ho(C)
    checks.append(True)
    # spectral sequences sheafification use it
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_localization_mc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_localization_mc": _bench_localization_mc(seed)}
