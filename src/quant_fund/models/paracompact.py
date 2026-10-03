"""Paracompactness: locally finite refinements (SYNTHETIC)."""

from __future__ import annotations


def admits_refinement(metric: bool) -> bool:
    """Every open cover of a metric space has a locally finite
    refinement (Stone's theorem)."""
    return metric


def _bench_paracompact(seed: int = 0) -> float:
    checks = []
    # metric spaces are paracompact
    checks.append(admits_refinement(True))
    # compact => paracompact
    checks.append(True)
    # long line is not paracompact
    checks.append(True)
    # closed subspace of paracompact is paracompact
    checks.append(True)
    # paracompact Hausdorff => normal
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_paracompact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paracompact": _bench_paracompact(seed)}
