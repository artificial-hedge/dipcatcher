"""Los-Vaught completeness test (SYNTHETIC)."""

from __future__ import annotations


def los_vaught(aleph0_categorical: bool, has_finite_model: bool) -> bool:
    """If T is countably categorical and has no finite models,
    T is complete."""
    return aleph0_categorical and not has_finite_model


def _bench_vaught_test(seed: int = 0) -> float:
    checks = []
    # DLO is aleph0-categorical, no finite models -> complete
    checks.append(los_vaught(True, False))
    # theory of a finite structure is not covered by Vaught
    checks.append(not los_vaught(False, True))
    # categorical but with finite models -> cannot conclude
    checks.append(not los_vaught(True, True))
    # incomplete + non-categorical
    checks.append(not los_vaught(False, False))
    return float(sum(checks) / len(checks))


def bench_vaught_test(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vaught_test": _bench_vaught_test(seed)}
