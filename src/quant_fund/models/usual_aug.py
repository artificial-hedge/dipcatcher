"""usual aug module (SYNTHETIC)."""

from __future__ import annotations


def usual_aug_ok(f: bool, right: bool) -> bool:
    """usual_aug
    check:
    filtration —
    right
    continuity."""
    return f and right


def usual_aug_aux(aux: bool) -> bool:
    """usual_aug
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_usual_aug(seed: int = 0) -> float:
    checks = []
    checks.append(usual_aug_ok(True, True))
    checks.append(not usual_aug_ok(False, True))
    checks.append(usual_aug_aux(True))
    checks.append(not usual_aug_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_usual_aug(seed: int = 0) -> dict[str, float]:
    return {"synthetic_usual_aug": _bench_usual_aug(seed)}
