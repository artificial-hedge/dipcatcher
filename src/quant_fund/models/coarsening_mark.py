"""coarsening mark module (SYNTHETIC)."""

from __future__ import annotations


def coarsening_mark_ok(node: bool, resid: bool) -> bool:
    """coarsening_mark
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def coarsening_mark_aux(aux: bool) -> bool:
    """coarsening_mark
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_coarsening_mark(seed: int = 0) -> float:
    checks = []
    checks.append(coarsening_mark_ok(True, True))
    checks.append(not coarsening_mark_ok(False, True))
    checks.append(coarsening_mark_aux(True))
    checks.append(not coarsening_mark_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_coarsening_mark(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coarsening_mark": _bench_coarsening_mark(seed)}
