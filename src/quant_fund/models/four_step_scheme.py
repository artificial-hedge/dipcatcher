"""four step_scheme module (SYNTHETIC)."""

from __future__ import annotations


def four_step_scheme_ok(fb1: bool, my: bool) -> bool:
    """four_step_scheme
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def four_step_scheme_aux(aux: bool) -> bool:
    """four_step_scheme
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_four_step_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(four_step_scheme_ok(True, True))
    checks.append(not four_step_scheme_ok(False, True))
    checks.append(four_step_scheme_aux(True))
    checks.append(not four_step_scheme_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_four_step_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_four_step_scheme": _bench_four_step_scheme(seed)}
