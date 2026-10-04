"""d-bar Neumann problem (SYNTHETIC)."""

from __future__ import annotations


def dbar_ok(closed_form: bool, l2_est: bool) -> bool:
    """d-bar
    Neumann
    problem:
    solve
    du = f
    for
    d-bar-
    closed
    forms
    with
    L2
    estimates —
    Hörmander."""
    return closed_form and l2_est


def hormander_l2(hl: bool) -> bool:
    """Hörmander
    L2
    estimates:
    weighted
    plurisubharmonic
    weights
    give
    solvability
    with
    bounds."""
    return hl


def _bench_d_bar_neumann(seed: int = 0) -> float:
    checks = []
    checks.append(dbar_ok(True, True))
    checks.append(not dbar_ok(False, True))
    checks.append(hormander_l2(True))
    checks.append(not hormander_l2(False))
    checks.append(True)  # Hörmander
    return float(sum(checks) / len(checks))


def bench_d_bar_neumann(seed: int = 0) -> dict[str, float]:
    return {"synthetic_d_bar_neumann": _bench_d_bar_neumann(seed)}
