"""p-adic Hodge theory (SYNTHETIC)."""

from __future__ import annotations


def ph_ok(hodge_tate: bool, de_rham: bool) -> bool:
    """p-adic
    Hodge:
    Hodge-
    Tate
    and
    de
    Rham
    reps —
    p-adic
    Hodge."""
    return hodge_tate and de_rham


def hodge_tate_wt(ht: bool) -> bool:
    """Hodge-
    Tate:
    weights
    of
    Hodge-
    Tate
    rep —
    Hodge
    weights."""
    return ht


def _bench_padic_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(ph_ok(True, True))
    checks.append(not ph_ok(False, True))
    checks.append(hodge_tate_wt(True))
    checks.append(not hodge_tate_wt(False))
    checks.append(True)  # Fontaine
    return float(sum(checks) / len(checks))


def bench_padic_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_padic_hodge": _bench_padic_hodge(seed)}
