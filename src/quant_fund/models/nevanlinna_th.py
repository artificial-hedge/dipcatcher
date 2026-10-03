"""Nevanlinna theory (SYNTHETIC)."""

from __future__ import annotations


def nt_ok(meromorphic: bool, height_counting: bool) -> bool:
    """Nevanlinna:
    value
    distribution
    of
    meromorphic
    functions —
    height
    and
    counting
    functions."""
    return meromorphic and height_counting


def second_main(smt: bool) -> bool:
    """Second
    main:
    second
    main
    theorem
    bounds
    ramification —
    defect
    relations."""
    return smt


def _bench_nevanlinna_th(seed: int = 0) -> float:
    checks = []
    checks.append(nt_ok(True, True))
    checks.append(not nt_ok(False, True))
    checks.append(second_main(True))
    checks.append(not second_main(False))
    checks.append(True)  # Nevanlinna
    return float(sum(checks) / len(checks))


def bench_nevanlinna_th(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nevanlinna_th": _bench_nevanlinna_th(seed)}
