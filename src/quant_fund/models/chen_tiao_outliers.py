"""Chen-Liu-Tiao additive/innovational/level-shift outlier detection.

References
----------
- Chen, C. & Liu, L.-M. (1993). "Joint Estimation of Model
  Parameters and Outlier Effects in Time Series." *JASA*
  88(421), 284-297.
- Chang, I., Tiao, G.C. & Chen, C. (1988). "Estimation of
  Time Series Parameters in the Presence of Outliers."
  *Technometrics* 30(2), 193-204.
- Tsay, R.S. (1988). "Outliers, Level Shifts, and Variance
  Changes in Time Series." *Journal of Forecasting* 7(1),
  1-20.
- Balke, N.S. (1993). "Detecting Level Shifts in Time
  Series." *JBES* 11(1), 81-92.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The Chen-Tiao battery separates four intervention types on
an AR residual stream: additive outlier (AO — one bad
point), innovational outlier (IO — one bad innovation that
propagates through the AR filter), level shift (LS — a
permanent mean step), and transient change (TC — a step
that decays geometrically at rate delta). The honest piece
is *joint iterative estimation*: a single pass over a
contaminated series leaves the AR fit biased by the
outliers themselves, diluting every test statistic. We
therefore loop: fit AR(p), build each type's regressor
(AO: raw pulse; IO/LS/TC: indicator filtered through the
AR operator ``pi^{-1}(B)``), scan tau for the max-|t|
(type, tau) pair, subtract the estimated omega * regressor
from the adjusted series, refit, and rescan — until
nothing exceeds the Bonferroni-style threshold or
``max_rounds`` is hit. The threshold uses
``z_{alpha / (2 * n * 4)}`` (4 types over n candidates),
which is what keeps the clean control honest — a naive
3.5 cutoff produces false IO calls on uncontaminated data.
Guards: interior-only scan (p+2..n-2), dedup hits within
3 indices per type, degenerate near-zero regressor norm
skipped. ``synth_outliers`` plants one AO and one LS into
an AR(2); the bench gates on detection at the planted
taus with correct types and zero calls on clean data.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _ar_fit(x: FloatArray, p: int) -> FloatArray:
    n = x.size
    y = x[p:]
    z = np.column_stack([x[p - k - 1 : n - k - 1] for k in range(p)])
    b = np.linalg.solve(z.T @ z + 1e-10 * np.eye(p), z.T @ y)
    return np.asarray(b, dtype=np.float64)


def _resid_regressor(
    n: int,
    tau: int,
    kind: str,
    phi: FloatArray,
    delta: float = 0.7,
) -> FloatArray:
    """Intervention regressor in residual space: pi(B) * indicator."""
    ind = np.zeros(n)
    if kind == "IO":
        ind[tau] = 1.0
        return ind
    if kind == "AO":
        ind[tau] = 1.0
    elif kind == "LS":
        ind[tau:] = 1.0
    else:  # TC
        for t in range(tau, n):
            ind[t] = delta ** (t - tau)
    w = ind.copy()
    p = phi.size
    for k in range(1, p + 1):
        w[k:] -= phi[k - 1] * ind[k - 1 : n - 1]
    return w


def _series_correction(
    n: int,
    tau: int,
    kind: str,
    phi: FloatArray,
    delta: float = 0.7,
) -> FloatArray:
    """Series-space effect of the intervention (for subtraction)."""
    out = np.zeros(n)
    if kind == "AO":
        out[tau] = 1.0
        return out
    if kind == "LS":
        out[tau:] = 1.0
        return out
    if kind == "TC":
        for t in range(tau, n):
            out[t] = delta ** (t - tau)
        return out
    # IO: psi(B) pulse — propagates through AR
    p = phi.size
    for t in range(tau, n):
        val = 1.0 if t == tau else 0.0
        for k in range(1, p + 1):
            if t - k >= tau:
                val += phi[k - 1] * out[t - k]
        out[t] = val
    return out


def outlier_scan(
    x: FloatArray,
    p: int = 2,
    alpha: float = 0.05,
    types: tuple[str, ...] = ("AO", "IO", "LS", "TC"),
    max_rounds: int = 8,
) -> dict[str, object]:
    """Chen-Tiao battery with joint iterative estimation."""
    v = _as_series(x)
    n = v.size
    z_thr = float(norm.ppf(1.0 - alpha / (2.0 * n * len(types))))
    adjusted = v - np.mean(v)
    hits: list[dict[str, object]] = []
    lo, hi = p + 2, n - 2
    for _ in range(max_rounds):
        phi = _ar_fit(adjusted, p)
        resid = adjusted.copy()
        for k in range(1, p + 1):
            resid[k:] -= phi[k - 1] * adjusted[k - 1 : n - 1]
        # MAD-robust sigma: the unflagged outliers themselves
        # would inflate std and dilute every t-statistic
        med = float(np.median(resid[p:]))
        sigma = float(1.4826 * np.median(np.abs(resid[p:] - med)))
        sigma = max(sigma, 1e-8)
        best: tuple[float, str, int] | None = None
        for kind in types:
            for tau in range(lo, hi):
                w = _resid_regressor(n, tau, kind, phi)
                seg = w[p:]
                nrm = float(np.sum(seg**2))
                if nrm < 1e-10:
                    continue
                num = float(np.sum(resid[p:] * seg))
                t_stat = num / max(np.sqrt(sigma**2 * nrm), 1e-12)
                if best is None or abs(t_stat) > abs(best[0]):
                    best = (t_stat, kind, tau)
        if best is None or abs(best[0]) <= z_thr:
            break
        t_stat, kind, tau = best
        w = _resid_regressor(n, tau, kind, phi)
        seg = w[p:]
        omega = float(np.sum(resid[p:] * seg) / np.sum(seg**2))
        # dedupe: same type within 3 indices absorbs the neighbor
        dup = any(h["type"] == kind and abs(cast(int, h["tau"]) - tau) <= 3 for h in hits)
        if not dup:
            hits.append(
                {
                    "type": kind,
                    "tau": tau,
                    "t_stat": float(t_stat),
                    "omega_hat": omega,
                }
            )
        adjusted = adjusted - omega * _series_correction(n, tau, kind, phi)
    out: dict[str, object] = {
        "hits": hits,
        "threshold": z_thr,
    }
    return out


def synth_outliers(
    seed: int = 20261231 + 357,
    n: int = 160,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC AR(2) with planted AO + LS; clean control."""
    rng = np.random.default_rng(seed)
    phi = np.array([0.5, -0.2])
    eps = 0.3 * rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = phi[0] * y[t - 1] + phi[1] * y[t - 2] + eps[t]
    y = y.copy()
    y[n // 3] += 2.4  # additive outlier
    y[2 * n // 3 :] += 2.0  # level shift
    clean = np.zeros(n)
    for t in range(2, n):
        clean[t] = phi[0] * clean[t - 1] + phi[1] * clean[t - 2] + eps[t]
    return y.astype(np.float64), clean.astype(np.float64)


def bench_outliers(seed: int = 20261231 + 357) -> dict[str, float]:
    y, clean = synth_outliers(seed=seed)
    n = y.size
    r = outlier_scan(y)
    hits = cast(list[dict[str, object]], r["hits"])
    ao_hit = [h for h in hits if h["type"] == "AO" and abs(cast(int, h["tau"]) - n // 3) <= 3]
    ls_hit = [
        h for h in hits if h["type"] in ("LS", "TC") and abs(cast(int, h["tau"]) - 2 * n // 3) <= 4
    ]
    rc = outlier_scan(clean)
    clean_hits = len(cast(list[dict[str, object]], rc["hits"]))
    ok = len(ao_hit) > 0 and len(ls_hit) > 0 and clean_hits == 0
    out: dict[str, float] = {
        "synthetic_ao_tau_err": float(
            min((abs(cast(int, h["tau"]) - n // 3) for h in ao_hit), default=99)
        ),
        "synthetic_ls_tau_err": float(
            min((abs(cast(int, h["tau"]) - 2 * n // 3) for h in ls_hit), default=99)
        ),
        "synthetic_n_hits": float(len(hits)),
        "synthetic_clean_hits": float(clean_hits),
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
