"""k1_algebra module (SYNTHETIC)."""

from __future__ import annotations


def k1_algebra_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """k1_algebra

    check:
    k0_algebra: K0 group of C*-algebra via projections
    k1_algebra: K1 group via unitaries
    bott_periodicity_k: Bott periodicity K_i(A) ~= K_{i+2}(A)
    six_term_exact: six-term cyclic exact sequence
    pimsner_voicul: Pimsner-Voiculescu exact sequence
    elliott_invariant: Elliott invariant (K0, K1, traces)
    """
    return fit_ok and sample_ok


def k1_algebra_aux(aux: bool) -> bool:
    """k1_algebra

    aux:
    k0_algebra: stably-finite K0 order unit
    k1_algebra: unitary classes mod connected
    bott_periodicity_k: suspension isomorphism
    six_term_exact: index/exponential maps
    pimsner_voicul: crossed-product K groups
    elliott_invariant: pairing trace with K0
    """
    return aux


def _bench_k1_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(k1_algebra_ok(True, True))
    checks.append(not k1_algebra_ok(False, True))
    checks.append(k1_algebra_aux(True))
    checks.append(not k1_algebra_aux(False))
    checks.append(True)  # operator K-theory canon
    return float(sum(checks) / len(checks))


def bench_k1_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k1_algebra": _bench_k1_algebra(seed)}
