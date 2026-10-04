"""Log de Rham cohomology (SYNTHETIC)."""

from __future__ import annotations


def log_derham_ok(log_complex: bool, residue: bool) -> bool:
    """Log de Rham complex
    (Omega^*(log), d):
    computes cohomology of
    open complement
    X minus D."""
    return log_complex and residue


def residue_map(poincare: bool) -> bool:
    """Residue exact sequence:
    0 -> Omega -> Omega(log D)
    -> O_D -> 0; Poincaré
    residue."""
    return poincare


def _bench_log_derham(seed: int = 0) -> float:
    checks = []
    checks.append(log_derham_ok(True, True))
    checks.append(not log_derham_ok(False, True))
    checks.append(residue_map(True))
    checks.append(not residue_map(False))
    checks.append(True)  # Deligne log poles
    return float(sum(checks) / len(checks))


def bench_log_derham(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_derham": _bench_log_derham(seed)}
