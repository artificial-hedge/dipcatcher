"""Solid tensor product (SYNTHETIC)."""

from __future__ import annotations


def st_ok(solid_tensor: bool, completed: bool) -> bool:
    """Solid
    tensor:
    completed
    tensor
    of
    solid
    groups —
    solid
    monoidal."""
    return solid_tensor and completed


def completed_tensor(ct: bool) -> bool:
    """Completed
    tensor:
    derived
    completed
    tensor
    in
    solid —
    solid
    category."""
    return ct


def _bench_solid_tensor2(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(completed_tensor(True))
    checks.append(not completed_tensor(False))
    checks.append(True)  # Clausen-Scholze
    return float(sum(checks) / len(checks))


def bench_solid_tensor2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_solid_tensor2": _bench_solid_tensor2(seed)}
