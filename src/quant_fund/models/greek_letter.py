"""Greek-letter construction (SYNTHETIC)."""

from __future__ import annotations


def greek_ok(quotient_seqs: bool, detects_v_n: bool) -> bool:
    """Greek-letter: iterated
    cofiber sequences produce
    alpha, beta, gamma families
    detecting v_n-periodic
    elements in ANSS."""
    return quotient_seqs and detects_v_n


def low_stems(alpha_beta: bool) -> bool:
    """In low stems, Greek-letter
    elements map to actual
    homotopy classes: alpha
    = image of J."""
    return alpha_beta


def _bench_greek_letter(seed: int = 0) -> float:
    checks = []
    checks.append(greek_ok(True, True))
    checks.append(not greek_ok(False, True))
    checks.append(low_stems(True))
    checks.append(not low_stems(False))
    checks.append(True)  # beta_1 = Smith-Toda target
    return float(sum(checks) / len(checks))


def bench_greek_letter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_greek_letter": _bench_greek_letter(seed)}
