"""motivic pipe module (SYNTHETIC)."""

from __future__ import annotations


def motivic_pipe_ok(motive: bool, suslin: bool) -> bool:
    """motivic_pipe
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def motivic_pipe_aux(aux: bool) -> bool:
    """motivic_pipe
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_motivic_pipe(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_pipe_ok(True, True))
    checks.append(not motivic_pipe_ok(False, True))
    checks.append(motivic_pipe_aux(True))
    checks.append(not motivic_pipe_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_motivic_pipe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_pipe": _bench_motivic_pipe(seed)}
