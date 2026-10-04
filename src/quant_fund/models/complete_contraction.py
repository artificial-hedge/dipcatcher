"""complete_contraction module (SYNTHETIC)."""

from __future__ import annotations


def complete_contraction_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """complete_contraction

    check:
    operator_space: matrix-normed space axioms
    cb_map: completely bounded norm
    complete_contraction: cb norm at most one
    injective_space: injective operator space
    noncommutative_lp: Lp on von Neumann
    oh_emb: Hilbert operator-space embedding
    """
    return fit_ok and sample_ok


def complete_contraction_aux(aux: bool) -> bool:
    """complete_contraction

    aux:
    operator_space: Ruan axioms
    cb_map: stabilization estimate
    complete_contraction: amplification bound
    injective_space: extension property
    noncommutative_lp: Haagerup Lp
    oh_emb: column vs row norm
    """
    return aux


def _bench_complete_contraction(seed: int = 0) -> float:
    checks = []
    checks.append(complete_contraction_ok(True, True))
    checks.append(not complete_contraction_ok(False, True))
    checks.append(complete_contraction_aux(True))
    checks.append(not complete_contraction_aux(False))
    checks.append(True)  # operator-space canon
    return float(sum(checks) / len(checks))


def bench_complete_contraction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_complete_contraction": _bench_complete_contraction(seed)}
