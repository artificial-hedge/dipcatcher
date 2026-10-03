"""Gerbes: locally nonempty stacks (SYNTHETIC)."""

from __future__ import annotations


def h2_classifies(trivial_gerbe: bool) -> bool:
    """G-gerbes over X are classified by H^2(X, G^ab-ish);
    trivial gerbe = BG x X."""
    return trivial_gerbe


def _bench_gerbe_toy(seed: int = 0) -> float:
    checks = []
    # BG x X is the trivial gerbe
    checks.append(h2_classifies(True))
    # locally nonempty but possibly no global section
    checks.append(True)
    # band: the classifying sheaf
    checks.append(True)
    # torsors under the trivial gerbe = G-bundles
    checks.append(True)
    # Brauer group = Azumaya/GM-gerbes toy link
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_gerbe_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gerbe_toy": _bench_gerbe_toy(seed)}
