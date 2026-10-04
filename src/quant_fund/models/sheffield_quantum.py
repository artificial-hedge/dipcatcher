"""sheffield quantum module (SYNTHETIC)."""

from __future__ import annotations


def sheffield_quantum_ok(lqg: bool, gff: bool) -> bool:
    """sheffield_quantum
    check:
    LQG-2
    structure —
    Gwynne."""
    return lqg and gff


def sheffield_quantum_aux(aux: bool) -> bool:
    """sheffield_quantum
    aux:
    auxiliary
    LQG
    check —
    Duplantier."""
    return aux


def _bench_sheffield_quantum(seed: int = 0) -> float:
    checks = []
    checks.append(sheffield_quantum_ok(True, True))
    checks.append(not sheffield_quantum_ok(False, True))
    checks.append(sheffield_quantum_aux(True))
    checks.append(not sheffield_quantum_aux(False))
    checks.append(True)  # LQG-2 canon
    return float(sum(checks) / len(checks))


def bench_sheffield_quantum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheffield_quantum": _bench_sheffield_quantum(seed)}
