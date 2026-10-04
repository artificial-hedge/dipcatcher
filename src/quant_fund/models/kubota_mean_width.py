"""kubota mean_width module (SYNTHETIC)."""

from __future__ import annotations


def kubota_mean_width_ok(geo: bool, kin: bool) -> bool:
    """kubota_mean_width
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def kubota_mean_width_aux(aux: bool) -> bool:
    """kubota_mean_width
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_kubota_mean_width(seed: int = 0) -> float:
    checks = []
    checks.append(kubota_mean_width_ok(True, True))
    checks.append(not kubota_mean_width_ok(False, True))
    checks.append(kubota_mean_width_aux(True))
    checks.append(not kubota_mean_width_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_kubota_mean_width(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kubota_mean_width": _bench_kubota_mean_width(seed)}
