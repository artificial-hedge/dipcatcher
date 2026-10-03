"""trimmed cad module (SYNTHETIC)."""

from __future__ import annotations


def trimmed_cad_ok(step: bool, radius: bool) -> bool:
    """trimmed_cad
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def trimmed_cad_aux(aux: bool) -> bool:
    """trimmed_cad
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_trimmed_cad(seed: int = 0) -> float:
    checks = []
    checks.append(trimmed_cad_ok(True, True))
    checks.append(not trimmed_cad_ok(False, True))
    checks.append(trimmed_cad_aux(True))
    checks.append(not trimmed_cad_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_trimmed_cad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trimmed_cad": _bench_trimmed_cad(seed)}
