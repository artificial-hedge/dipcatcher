"""Anytime-valid e-processes: coverage misses + forecast loss differentials.

1. Bernoulli miss e-process vs nominal α (Ramdas / Grünwald / Shafer–Vovk;
   ADR-010). Ville: P(sup E_t ≥ 1/level) ≤ level with no peeking correction.
2. Loss-differential e-process (Choe–Ramdas style) for H0: E[d] ≤ 0 where
   d = L_A − L_B (smaller loss better). Research-diagnostic only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_EPS = 1e-12
_E_MAX = 1e300
_ERR_LEVEL = 0.05


def _clip_prob(p: float, name: str) -> float:
    x = float(p)
    if not 0.0 < x < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return float(np.clip(x, _EPS, 1.0 - _EPS))


def _alt_lambda(alpha: float) -> float:
    """Fixed alternative miss rate λ ∈ (0, 1), predictable from α only.

    λ = 2α detects undercoverage (too many misses). If 2α would leave
    (0, 1), fall back to the midpoint between α and 1.
    """
    a = float(alpha)
    lam = 2.0 * a
    if lam >= 1.0 - _EPS:
        lam = 0.5 * (a + 1.0)
    return float(np.clip(lam, _EPS, 1.0 - _EPS))


def _step_e(miss: Array, alpha: float) -> Array:
    a = _clip_prob(alpha, "alpha")
    lam = _alt_lambda(a)
    x = np.clip(np.asarray(miss, dtype=float), 0.0, 1.0)
    raw = (x / a) * lam + ((1.0 - x) / (1.0 - a)) * (1.0 - lam)
    return np.asarray(np.clip(raw, 0.0, _E_MAX), dtype=np.float64)


def e_value_bernoulli(miss: float, alpha: float) -> float:
    """One-step e-value for a miss indicator versus expected miss α.

    e = (miss / α) λ + ((1 − miss) / (1 − α)) (1 − λ),  λ = 2α (capped).

    Honesty note: ``miss`` is silently clipped to ``[0, 1]`` inside ``_step_e``
    (soft coverage / fractional miss allowed). Out-of-range values are *not*
    rejected — callers that need hard {0,1} indicators must pre-validate.
    ``bench_e_coverage`` likewise clips covered∈[0,1] before forming misses.
    """
    return float(_step_e(np.asarray([miss], dtype=float), alpha)[0])


def e_process(misses: Array, alpha: float) -> Array:
    """Running product E_t = ∏_{s≤t} e_s. Implicit E_0 = 1. Anytime-valid."""
    x = np.asarray(misses, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([1.0], dtype=float)
    steps = _step_e(x, alpha)
    log_steps = np.log(np.clip(steps, _EPS, _E_MAX))
    log_run = np.minimum(np.cumsum(log_steps), np.log(_E_MAX))
    return np.asarray(np.exp(log_run), dtype=np.float64)


def e_process_threshold(e: Array, level: float = 0.05) -> dict[str, object]:
    """First time the e-process meets 1/level (Ville). No peeking penalty."""
    path = np.asarray(e, dtype=float).reshape(-1)
    thresh = 1.0 / _clip_prob(level, "level")
    hit = path >= thresh
    if path.size > 0 and bool(np.any(hit)):
        first_cross: int | None = int(np.argmax(hit))
        reject = True
    else:
        first_cross = None
        reject = False
    return {
        "first_cross": first_cross,
        "e": path,
        "reject": reject,
        "threshold": float(thresh),
    }


def bench_e_coverage(covered: Array, alpha: float = 0.10) -> dict[str, object]:
    """Empirical coverage plus the anytime-valid miss e-process (level 0.05)."""
    c = np.asarray(covered, dtype=float).reshape(-1)
    c = c[np.isfinite(c)]
    n = int(c.size)
    if n == 0:
        return {"coverage": float("nan"), "e_final": 1.0, "ever_cross": False, "n": 0}
    misses = 1.0 - np.clip(c, 0.0, 1.0)
    path = e_process(misses, alpha)
    decision = e_process_threshold(path, level=_ERR_LEVEL)
    return {
        "coverage": float(np.mean(c)),
        "e_final": float(path[-1]),
        "ever_cross": bool(decision["reject"]),
        "n": n,
    }


def _clip_lam(lam: float) -> float:
    x = float(lam)
    if not 0.0 < x < 1.0:
        raise ValueError("lam must be in (0, 1)")
    return float(np.clip(x, _EPS, 1.0 - _EPS))


def _align_loss_diff(
    loss_a: Array | None,
    loss_b: Array | None,
    d: Array | None,
) -> Array:
    """Build d_t = L_A,t − L_B,t (smaller loss better). Finite pairs only."""
    if d is not None:
        if loss_a is not None or loss_b is not None:
            raise ValueError("pass either d or (loss_a, loss_b), not both")
        diff = np.asarray(d, dtype=float).reshape(-1)
    else:
        if loss_a is None or loss_b is None:
            raise ValueError("need loss_a and loss_b, or d")
        a = np.asarray(loss_a, dtype=float).reshape(-1)
        b = np.asarray(loss_b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("loss series must align")
        diff = a - b
    return diff[np.isfinite(diff)]


def _validate_initial_bound(initial_bound: float) -> float:
    b0 = float(initial_bound)
    if not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("initial_bound must be finite and > 0")
    return b0


def e_process_loss_diff(
    loss_a: Array | None = None,
    loss_b: Array | None = None,
    *,
    d: Array | None = None,
    lam: float = 0.25,
    initial_bound: float = 1.0,
) -> Array:
    """Anytime-valid capital process for the median-null ``median(d) ≤ 0``.

    ``d_t = L_{A,t} − L_{B,t}`` (smaller loss better). Rejecting / wealth
    growth is evidence that A is systematically **worse** than B.

    Construction (sign bet — the same distribution-free construction as
    ``research.evalues.LossEProcess``):
    1. Align losses → finite ``d``.
    2. One-step factor ``e_t = 1 + lam_t·sign(d_t)`` with
       ``lam_t = clip(2·p̂_t − 1, 0, lam)``, ``p̂_t`` the Laplace-smoothed
       running win-rate computed on strict history (predictable, and
       clipped nonnegative — all validity needs under the null).
    3. Capital ``E_t = ∏_{s≤t} e_s`` (log-cumsum; clipped like coverage).

    Why the sign bet: the earlier bounded-exponential factor
    ``exp(λ·clip(d_t/b_t, −1, 1) − ψ_E(λ)·clip²)`` was *not* a valid
    e-factor for the mean-null it claimed — clipping is asymmetric for
    skewed ``d`` (a stable left tail gets saturated at −1 while the
    positive side stays linear, so ``E[clip(d/b)] > 0`` under ``E[d] ≤ 0``),
    and the running bound ``b_t`` leaves a transient window before a first
    large ``|d|`` is seen where every factor has ``E[e_t] > 1``
    (meta-audit: 92% Ville-cross rate on a mean-0 two-point stream).
    No e-process can test the raw mean-null ``E[d] ≤ 0`` for unbounded
    differentials, so the process now tests the strongest distribution-free
    notion of "not worse" — ``median(d) ≤ 0`` ⇒ ``E[sign(d)] ≤ 0`` ⇒
    ``E[e_t | F_{t-1}] ≤ 1`` under arbitrary tails, skew, scale, and
    misspecification. ``lam_t`` adaptivity only trades power inside the
    valid envelope, never validity.

    ``initial_bound`` is retained for API compatibility and is unused by
    the bet — the sign bet needs no scale. Empty → ``[1.0]``.
    """
    _validate_initial_bound(initial_bound)
    diff = _align_loss_diff(loss_a, loss_b, d)
    if diff.size == 0:
        return np.array([1.0], dtype=float)
    lam_cap = _clip_lam(lam)
    g = np.sign(diff)  # +1 on d>0 (A worse) — growth side of the bet
    # predictable lam_t: wins = #(d>0) among strict history
    wins_before = np.concatenate(([0.0], np.cumsum(g > 0.0)[:-1]))
    t_idx = np.arange(diff.size, dtype=float)
    p_hat = (wins_before + 1.0) / (t_idx + 2.0)  # Laplace pseudo-count 1
    lam_t = np.clip(2.0 * p_hat - 1.0, 0.0, lam_cap)
    e_t = 1.0 + lam_t * g  # in (1 - lam, 1 + lam), strictly positive
    log_run = np.minimum(np.cumsum(np.log(e_t)), np.log(_E_MAX))
    path: Array = np.exp(log_run).astype(float)
    return path


def e_process_dm(
    loss_a: Array,
    loss_b: Array,
    *,
    lam: float = 0.25,
    initial_bound: float = 1.0,
    level: float = 0.05,
) -> dict[str, object]:
    """DM-style wrapper: e-process on loss_a − loss_b plus Ville decision.

    Keys: ``e``, ``e_final``, ``mean_loss_diff``, ``n``, plus threshold fields
    from ``e_process_threshold`` (``first_cross``, ``reject``, ``threshold``).
    Research-diagnostic only.
    """
    a = np.asarray(loss_a, dtype=float).reshape(-1)
    b = np.asarray(loss_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("loss series must align")
    diff = a - b
    finite = np.isfinite(diff)
    d = diff[finite]
    n = int(d.size)
    path = e_process_loss_diff(d=d, lam=lam, initial_bound=initial_bound)
    decision = e_process_threshold(path, level=level)
    return {
        "e": path,
        "e_final": float(path[-1]),
        "mean_loss_diff": float(np.mean(d)) if n else float("nan"),
        "n": n,
        "first_cross": decision["first_cross"],
        "reject": bool(decision["reject"]),
        "threshold": float(decision["threshold"]),  # type: ignore[arg-type]
    }
