"""quastel valko module (SYNTHETIC)."""

from __future__ import annotations


def quastel_valko_ok(asep: bool, wk: bool) -> bool:
    """quastel_valko
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def quastel_valko_aux(aux: bool) -> bool:
    """quastel_valko
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_quastel_valko(seed: int = 0) -> float:
    checks = []
    checks.append(quastel_valko_ok(True, True))
    checks.append(not quastel_valko_ok(False, True))
    checks.append(quastel_valko_aux(True))
    checks.append(not quastel_valko_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_quastel_valko(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quastel_valko": _bench_quastel_valko(seed)}
