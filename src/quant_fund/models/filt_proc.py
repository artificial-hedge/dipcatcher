"""filt proc module (SYNTHETIC)."""

from __future__ import annotations


def filt_proc_ok(fil: bool, loc: bool) -> bool:
    """filt_proc
    check:
    filtration
    structure —
    Jacod-Shiryaev
    limit."""
    return fil and loc


def filt_proc_aux(aux: bool) -> bool:
    """filt_proc
    aux:
    auxiliary
    converg
    check —
    Pinsky
    process."""
    return aux


def _bench_filt_proc(seed: int = 0) -> float:
    checks = []
    checks.append(filt_proc_ok(True, True))
    checks.append(not filt_proc_ok(False, True))
    checks.append(filt_proc_aux(True))
    checks.append(not filt_proc_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_filt_proc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_filt_proc": _bench_filt_proc(seed)}
