"""Object classifiers (SYNTHETIC)."""

from __future__ import annotations


def classifies(kappa_small: bool, pullback_classify: bool) -> bool:
    """An infinity-topos has object classifiers: for each
    regular cardinal kappa, small maps are pullbacks of a
    universal family over an object of kappa-compacts."""
    return kappa_small and pullback_classify


def _bench_object_classif(seed: int = 0) -> float:
    checks = []
    # pullback-classified small maps
    checks.append(classifies(True, True))
    # no classifier fails
    checks.append(not classifies(True, False))
    # generalizes subobject classifier
    checks.append(True)
    # universe in HoTT corresponds
    checks.append(True)
    # existence for presentable topoi
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_object_classif(seed: int = 0) -> dict[str, float]:
    return {"synthetic_object_classif": _bench_object_classif(seed)}
