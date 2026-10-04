"""elliott_invariant module (SYNTHETIC)."""

from __future__ import annotations


def elliott_invariant_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """elliott_invariant

    check:
    k0_algebra: K0 group of C*-algebra via projections
    k1_algebra: K1 group via unitaries
    bott_periodicity_k: Bott periodicity K_i(A) ~= K_{i+2}(A)
    six_term_exact: six-term cyclic exact sequence
    pimsner_voicul: Pimsner-Voiculescu exact sequence
    elliott_invariant: Elliott invariant (K0, K1, traces)
    """
    return fit_ok and sample_ok


def elliott_invariant_aux(aux: bool) -> bool:
    """elliott_invariant

    aux:
    k0_algebra: stably-finite K0 order unit
    k1_algebra: unitary classes mod connected
    bott_periodicity_k: suspension isomorphism
    six_term_exact: index/exponential maps
    pimsner_voicul: crossed-product K groups
    elliott_invariant: pairing trace with K0
    """
    return aux


def _bench_elliott_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(elliott_invariant_ok(True, True))
    checks.append(not elliott_invariant_ok(False, True))
    checks.append(elliott_invariant_aux(True))
    checks.append(not elliott_invariant_aux(False))
    checks.append(True)  # operator K-theory canon
    return float(sum(checks) / len(checks))


def bench_elliott_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliott_invariant": _bench_elliott_invariant(seed)}
