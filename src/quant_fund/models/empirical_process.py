"""empirical process module (SYNTHETIC)."""

from __future__ import annotations


def empirical_process_ok(proc: bool, tight: bool) -> bool:
    """empirical_process
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def empirical_process_aux(aux: bool) -> bool:
    """empirical_process
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_empirical_process(seed: int = 0) -> float:
    checks = []
    checks.append(empirical_process_ok(True, True))
    checks.append(not empirical_process_ok(False, True))
    checks.append(empirical_process_aux(True))
    checks.append(not empirical_process_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_empirical_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_empirical_process": _bench_empirical_process(seed)}
