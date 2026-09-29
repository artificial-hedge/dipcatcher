"""E-value mergers — pooling evidence across heads, shards, and lanes.

The sequential-inference suite produces one e-value per lane
(promotion, coverage, calibration, drift, tail depth, localization).
This module provides the **only** valid ways to combine them, per
Vovk & Wang (2020) "Combining e-values" and the follow-up literature:

- ``emerge_mean`` — the arithmetic mean of e-values is ALWAYS an
  e-value, under arbitrary dependence between them. The universal,
  assumption-free merger.
- ``emerge_harmonic`` — the harmonic mean is an e-value under arbitrary
  dependence (Vovk & Wang 2020, Thm. 10 — harmonic mean of e-values
  multiplied by ln(k+1)... we use the plain harmonic mean which is
  conservative-valid: 1/(k * mean(1/e)) bounds are looser; the exact
  statement we rely on is that the harmonic mean times any constant
  <= 1 is an e-value — harmonic is always <= arithmetic so it stays
  valid, just conservative).
- ``emerge_product`` — the product is an e-value ONLY when the inputs
  are independent (or mutually independent conditional on the null).
  Flagged loudly: use only for receipts built on disjoint data.
- ``emerge_bonferroni`` — min(e_i) * k is an e-value under arbitrary
  dependence (Bonferroni correction). Best when a single lane carries
  most of the evidence; worst when evidence is diffuse.
- ``emerge_simes`` — max_i (k * e_(i) / (i * H_k)) is an e-value under
  arbitrary dependence (the e-value analog of Simes' rule with the
  harmonic correction H_k = sum_j 1/j, Vovk & Wang). The unadjusted
  form k * e_(i) / i is only valid under independence — we ship the
  corrected version.

p-to-e and e-to-p calibrators (Shafer et al. / Vovk & Wang):
- ``p_to_e``: f(p) = p^{v-1} for v in (0,1) is a valid p-to-e
  calibrator for ANY p-variable. We use v = 1/2: e = 1/sqrt(p).
- ``e_to_p``: p = min(1, 1/e) is an exact e-to-p calibrator
  (Markov/Ville bound — sharp, cannot be improved in general).

All mergers validate inputs: non-finite or negative e-values raise —
a NaN e-value must fail closed, never launder as evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "e_to_p",
    "emerge_bonferroni",
    "emerge_harmonic",
    "emerge_mean",
    "emerge_product",
    "emerge_simes",
    "p_to_e",
]


def _check_evals(evals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(evals), dtype=float)
    if arr.size == 0:
        raise ValueError("at least one e-value required")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("e-values must be finite and >= 0")
    return arr


def emerge_mean(evals: Sequence[float]) -> float:
    """Arithmetic mean — valid under ARBITRARY dependence."""
    return float(np.mean(_check_evals(evals)))


def emerge_harmonic(evals: Sequence[float]) -> float:
    """Harmonic mean — valid under arbitrary dependence (conservative).

    harmonic <= arithmetic pointwise, so validity follows from the mean
    merger; the price is conservativeness when evidence is uneven.
    Zero e-values make the harmonic mean exactly 0 (one lane vetoed the
    merged claim — correct semantics for evidence pooling).
    """
    arr = _check_evals(evals)
    if np.any(arr == 0.0):
        return 0.0
    return float(arr.size / np.sum(1.0 / arr))


def emerge_product(evals: Sequence[float]) -> float:
    """Product — valid ONLY for independent e-values.

    Caller asserts independence; there is no way to verify it from the
    e-values themselves. Independent means the evidence streams are
    built on disjoint data under the null.
    """
    arr = _check_evals(evals)
    return float(np.prod(arr))


def emerge_bonferroni(evals: Sequence[float]) -> float:
    """k * min(e_i) — the 'all must pass' merger, arbitrary dependence."""
    arr = _check_evals(evals)
    return float(arr.size * np.min(arr))


def emerge_simes(evals: Sequence[float]) -> float:
    """e-Simes merger: max_i (k * e_(i) / (i * H_k)).

    H_k = sum_{j<=k} 1/j is the harmonic correction making it valid
    under ARBITRARY dependence. Dominates Bonferroni; costs a
    ~ln(k) factor versus the independence-only unadjusted form.
    """
    arr = np.sort(_check_evals(evals))
    k = arr.size
    i = np.arange(1, k + 1)
    harmonic = float(np.sum(1.0 / i))
    return float(np.max(k * arr / (i * harmonic)))


def p_to_e(p: float, v: float = 0.5) -> float:
    """Calibrate a p-value to an e-value: e = p^(v-1), v in (0, 1).

    Valid for any p-variable (super-uniform under the null). Default
    v = 1/2 gives e = 1/sqrt(p). Loses the sequential-validity of a
    true e-process — this is a point statistic only.
    """
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if not (np.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError("v must be in (0, 1)")
    if p <= 0.0:
        return float("inf")
    return float(p ** (v - 1.0))


def e_to_p(e: float) -> float:
    """Calibrate an e-value to a p-value: p = min(1, 1/e).

    Sharp by the Markov inequality for e-variables — no uniformly
    better calibrator exists without more structure.
    """
    if not (np.isfinite(e) and e >= 0.0):
        raise ValueError("e must be finite and >= 0")
    if e <= 1.0:
        return 1.0
    return float(min(1.0, 1.0 / e))
