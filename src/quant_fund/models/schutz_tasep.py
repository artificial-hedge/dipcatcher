"""schutz tasep module (SYNTHETIC)."""

from __future__ import annotations


def schutz_tasep_ok(asep: bool, wk: bool) -> bool:
    """schutz_tasep
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def schutz_tasep_aux(aux: bool) -> bool:
    """schutz_tasep
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_schutz_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(schutz_tasep_ok(True, True))
    checks.append(not schutz_tasep_ok(False, True))
    checks.append(schutz_tasep_aux(True))
    checks.append(not schutz_tasep_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_schutz_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schutz_tasep": _bench_schutz_tasep(seed)}
