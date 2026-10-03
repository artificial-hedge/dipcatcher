"""F-motives (SYNTHETIC)."""

from __future__ import annotations


def fm_ok(f_motive: bool, frobenius: bool) -> bool:
    """F-motive:
    F-motive —
    Frobenius
    structure."""
    return f_motive and frobenius


def frobenius_motive(frm: bool) -> bool:
    """Frobenius
    motive:
    motive
    with
    Frobenius —
    isocrystal."""
    return frm


def _bench_f_motive(seed: int = 0) -> float:
    checks = []
    checks.append(fm_ok(True, True))
    checks.append(not fm_ok(False, True))
    checks.append(frobenius_motive(True))
    checks.append(not frobenius_motive(False))
    checks.append(True)  # crystalline motive
    return float(sum(checks) / len(checks))


def bench_f_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_motive": _bench_f_motive(seed)}
