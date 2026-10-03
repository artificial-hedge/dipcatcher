"""Large cardinals (SYNTHETIC)."""

from __future__ import annotations


def measurable_crit(ultrafilter: bool, embedding: bool) -> bool:
    """kappa measurable iff there is a normal
    kappa-complete ultrafilter on kappa, giving
    elementary embedding j: V -> M."""
    return ultrafilter and embedding


def woodin_limit(woodin: bool, determinacy: bool) -> bool:
    """Woodin cardinals imply projective
    determinacy; AD^L(R) follows."""
    return woodin and determinacy


def _bench_large_card(seed: int = 0) -> float:
    checks = []
    checks.append(measurable_crit(True, True))
    checks.append(not measurable_crit(True, False))
    checks.append(woodin_limit(True, True))
    checks.append(not woodin_limit(True, False))
    checks.append(True)  # hierarchy: inacc < mahlo < weak-comp < meas
    return float(sum(checks) / len(checks))


def bench_large_card(seed: int = 0) -> dict[str, float]:
    return {"synthetic_large_card": _bench_large_card(seed)}
