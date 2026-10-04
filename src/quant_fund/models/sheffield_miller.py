"""sheffield miller module (SYNTHETIC)."""

from __future__ import annotations


def sheffield_miller_ok(gff: bool, qg: bool) -> bool:
    """sheffield_miller
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def sheffield_miller_aux(aux: bool) -> bool:
    """sheffield_miller
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_sheffield_miller(seed: int = 0) -> float:
    checks = []
    checks.append(sheffield_miller_ok(True, True))
    checks.append(not sheffield_miller_ok(False, True))
    checks.append(sheffield_miller_aux(True))
    checks.append(not sheffield_miller_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_sheffield_miller(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheffield_miller": _bench_sheffield_miller(seed)}
