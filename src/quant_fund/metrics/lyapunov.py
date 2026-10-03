"""Nonlinear-dynamics diagnostics from a single observed time series.

Modules
-------
* ``embed_series`` — delay-coordinate embedding (Takens 1981): an observed
  scalar series ``x_t`` is reconstructed as vectors
  ``v_t = (x_t, x_{t+τ}, …, x_{t+(m−1)τ})`` which generically recover the
  attractor's topology for large enough ``m`` and suitable delay ``τ``.
* ``largest_lyapunov`` — Rosenstein et al. (1993) estimator of the largest
  Lyapunov exponent: pairs of nearby embedded states diverge as
  ``‖v_i(t) − v_j(t)‖ ≈ C_i e^{λ t}``; ``λ`` is the slope of the mean log
  divergence ``⟨ln d_j(t)⟩_j`` against time, restricted to the initial
  linear region before saturation.
* ``false_nearest_neighbors`` — Kennel–Brown–Abarbanel (1992) test for the
  minimum embedding dimension: when ``m`` is too small, projection creates
  false neighbors whose expansion ratio after adding the next coordinate
  exceeds a threshold; the FNN fraction should drop to ~0 at the true
  embedding dimension.
* ``cao_dimension`` — Cao (1997) criterion, a self-averaging variant of FNN
  that requires no threshold and works on deterministic and stochastic
  series: ``E1(m)`` saturates at the attractor dimension.

Honesty contract
----------------
* All inputs are observed or synthetic time series; no prices, NAVs, or
  P&L appear and no Sharpe/Sortino/Calmar-style summaries are produced.
* Lyapunov exponents characterize deterministic chaos; they are not
  forecasting statistics and are labeled ``synthetic_*`` in benches.
* Estimates are biased upward by noise and downward by embedding errors;
  reported values carry no uncertainty statement beyond the synthetic
  reference experiments in ``bench_lyapunov``.

Composition
-----------
* ``rqa.py`` supplies recurrence-based measures of the same embedded
  trajectories; this module is orthogonal (divergence, not recurrence).
* ``surrogate_nonlinear.py`` consumes λ estimates as a null-model
  comparison statistic for nonlinearity testing.

References
----------
* Takens, F. (1981), "Detecting Strange Attractors in Turbulence", in
  Dynamical Systems and Turbulence, LNM 898, Springer.
* Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993), "A Practical
  Method for Calculating Largest Lyapunov Exponents from Small Data
  Sets", Physica D 65:117–134.
* Kennel, M.B., Brown, R., Abarbanel, H.D.I. (1992), "Determining
  Embedding Dimension for Phase-Space Reconstruction Using a Geometrical
  Construction", Physical Review A 45:3403–3411.
* Cao, L. (1997), "Practical Method for Determining the Minimum Embedding
  Dimension of a Scalar Time Series", Physica D 110:43–50.
* Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985),
  "Determining Lyapunov Exponents from a Time Series", Physica D
  16:285–317.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

FloatArray = np.ndarray


def _as_series(x: FloatArray | list[float]) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 64:
        raise ValueError("series too short for dynamical diagnostics (need >= 64)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series must be finite")
    return arr


def embed_series(x: FloatArray, dim: int, tau: int = 1) -> FloatArray:
    """Delay embedding: (T, dim) matrix of lagged coordinates.

    Raises
    ------
    ValueError
        If ``dim < 1``, ``tau < 1``, or the series is too short.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if dim < 1 or tau < 1:
        raise ValueError("dim and tau must be >= 1")
    n_vec = x.size - (dim - 1) * tau
    if n_vec < 8:
        raise ValueError("series too short for requested embedding")
    idx = np.arange(dim) * tau + np.arange(n_vec)[:, None]
    return np.asarray(x[idx])


def _pair_divergence(emb: FloatArray, theiler: int, horizon: int) -> tuple[FloatArray, FloatArray]:
    """Mean log nearest-neighbor divergence for steps 0..horizon."""
    tree = cKDTree(emb)
    n = emb.shape[0]
    # nearest neighbor of each state, excluding a Theiler window
    dists, nn_idx = tree.query(emb, k=2 * theiler + 3, workers=-1)
    partner = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for cand in nn_idx[i]:
            if cand != i and abs(int(cand) - i) > theiler:
                partner[i] = int(cand)
                break
    valid = np.where((partner >= 0) & (partner < n - horizon) & (np.arange(n) < n - horizon))[0]
    if valid.size < 10:
        raise ValueError("embedding too small / theiler too large for divergence")
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    log_div = np.empty(horizon)
    for k in range(horizon):
        i = valid
        j = partner[valid]
        d = np.linalg.norm(emb[i + k + 1] - emb[j + k + 1], axis=1)
        d0 = np.linalg.norm(emb[i] - emb[j], axis=1)
        ratio = d / np.maximum(d0, 1e-12)
        log_div[k] = np.log(np.maximum(ratio, 1e-12)).mean()
    return steps, log_div


def largest_lyapunov(
    x: FloatArray,
    dim: int = 6,
    tau: int = 1,
    theiler: int | None = None,
    fit_steps: int = 8,
    dt: float = 1.0,
) -> dict[str, FloatArray | float]:
    """Rosenstein (1993) largest-Lyapunov estimate.

    Parameters
    ----------
    x
        Observed scalar series.
    dim
        Embedding dimension (should satisfy m ≥ 2d + 1 for attractor dim d
        in theory; Cao criterion picks it empirically).
    tau
        Delay in samples (first minimum of mutual information is the
        standard choice; see ``rqa.mutual_information_delay``).
    theiler
        Temporal exclusion window for neighbor search; defaults to the
        first lag where the series' autocorrelation falls below 1/e,
        bounded in [1, 20].
    fit_steps
        Number of early divergence steps whose mean slope is λ.
    dt
        Sample spacing in physical time; λ is returned per unit of it.

    Returns
    -------
    dict with keys ``lyap`` (slope), ``steps``, ``log_divergence``,
    ``n_pairs``, ``dim``, ``tau``, ``theiler``.
    """
    x = _as_series(x)
    if theiler is None:
        xc = x - x.mean()
        ac = np.correlate(xc, xc, "full")[x.size - 1 :] / max(xc.dot(xc), 1e-12)
        below = np.where(ac < math.exp(-1))[0]
        theiler = int(below[0]) if below.size else 1
        theiler = int(max(1, min(20, theiler)))
    horizon = max(fit_steps + 1, 3 * theiler + fit_steps)
    emb = embed_series(x, dim, tau)
    steps, log_div = _pair_divergence(emb, theiler, horizon)
    k = min(fit_steps, log_div.size)
    t = steps[:k] * dt
    y = log_div[:k]
    slope = float(np.polyfit(t, y, 1)[0])
    return {
        "lyap": slope,
        "steps": steps * dt,
        "log_divergence": log_div,
        "n_pairs": float(emb.shape[0]),
        "dim": float(dim),
        "tau": float(tau),
        "theiler": float(theiler),
    }


def false_nearest_neighbors(
    x: FloatArray,
    max_dim: int = 8,
    tau: int = 1,
    atol_ratio: float = 15.0,
    rtol: float = 2.0,
    theiler: int = 10,
) -> FloatArray:
    """Kennel et al. (1992) FNN fraction for dims 1..max_dim.

    A neighbor is "false" when the extra coordinate added at dim m+1
    separates the pair by more than ``rtol × A_t`` (attractor size) or the
    relative expansion ``(R_{m+1} − R_m) / R_m`` exceeds ``atol_ratio``.
    """
    x = _as_series(x)
    fnn = np.empty(max_dim)
    ra = float(np.std(x))
    if ra == 0.0:
        raise ValueError("constant series has no dynamics")
    for m in range(1, max_dim + 1):
        emb_m = embed_series(x, m, tau)
        emb_p = embed_series(x, m + 1, tau)
        n = min(emb_m.shape[0], emb_p.shape[0])
        emb_m = emb_m[:n]
        tree = cKDTree(emb_m)
        dist, idx = tree.query(emb_m, k=2, workers=-1)
        nn = idx[:, 1]
        dd = dist[:, 1]
        # enforce theiler window by re-querying neighbors for violators
        bad = np.abs(nn - np.arange(n)) <= theiler
        if bad.any():
            idx3 = tree.query(emb_m[bad], k=theiler * 2 + 4, workers=-1)[1]
            for i, row in zip(np.where(bad)[0], idx3, strict=True):
                ok = [c for c in row if c != i and abs(c - i) > theiler]
                if ok:
                    nn[i] = ok[0]
                    dd[i] = float(np.linalg.norm(emb_m[i] - emb_m[nn[i]]))
        r_next = np.linalg.norm(emb_p[:n] - emb_p[nn], axis=1)
        expand = (r_next - dd) / np.maximum(dd, 1e-12)
        false = (expand > atol_ratio) | (r_next / max(ra, 1e-12) > rtol)
        fnn[m - 1] = float(false.mean())
    return fnn


def cao_dimension(
    x: FloatArray, max_dim: int = 10, tau: int = 1, theiler: int = 10
) -> dict[str, FloatArray]:
    """Cao (1997) E1 curve: mean distance expansion per dim, saturating at
    the true embedding dimension. Returns ``e`` (per-dim mean expansion),
    ``e1`` (successive ratios), and ``dim_cao`` (first dim with e1 ≈ 1,
    or ``max_dim`` when unsaturated)."""
    x = _as_series(x)
    e = np.empty(max_dim)
    for m in range(1, max_dim + 1):
        emb_m = embed_series(x, m, tau)
        emb_p = embed_series(x, m + 1, tau)
        n = min(emb_m.shape[0], emb_p.shape[0])
        emb_m = emb_m[:n]
        tree = cKDTree(emb_m)
        idx = tree.query(emb_m, k=theiler * 2 + 4, workers=-1)[1]
        rat = np.empty(n)
        for i in range(n):
            cand = [c for c in idx[i] if c != i and abs(c - i) > theiler]
            if not cand:
                rat[i] = np.nan
                continue
            j = cand[0]
            d_m = float(np.linalg.norm(emb_m[i] - emb_m[j]))
            d_p = float(np.linalg.norm(emb_p[i] - emb_p[j]))
            rat[i] = d_p / max(d_m, 1e-12)
        e[m - 1] = float(np.nanmean(rat))
    e1 = np.empty(max_dim - 1)
    e1[:] = np.nan
    for m in range(max_dim - 1):
        if e[m] > 1e-12:
            e1[m] = e[m + 1] / e[m]
    sat = int(np.argmax(np.abs(e1 - 1.0) < 0.05)) if np.any(np.abs(e1 - 1.0) < 0.05) else -1
    chosen = sat + 1 if sat >= 0 else max_dim
    out: dict[str, FloatArray] = {"e": e, "e1": e1, "dim_cao": np.array(float(chosen))}
    return out


def synth_logistic(n: int = 4000, r: float = 4.0, seed: int = 0) -> FloatArray:
    """Logistic map x_{t+1} = r x_t (1 − x_t) at r = 4; true λ = ln 2 ≈ 0.693."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = float(rng.uniform(0.1, 0.9))
    for t in range(1, n):
        x[t] = r * x[t - 1] * (1.0 - x[t - 1])
    return np.clip(x, 1e-9, 1.0 - 1e-9)


def synth_henon(n: int = 4000, a: float = 1.4, b: float = 0.3, seed: int = 0) -> FloatArray:
    """Hénon map x-component; largest Lyapunov ≈ 0.419."""
    rng = np.random.default_rng(seed)
    xy = np.empty((n, 2))
    xy[0] = rng.uniform(-0.5, 0.5, 2)
    for t in range(1, n):
        xy[t, 0] = 1.0 - a * xy[t - 1, 0] ** 2 + xy[t - 1, 1]
        xy[t, 1] = b * xy[t - 1, 0]
    return xy[:, 0]


def synth_periodic(n: int = 4000, period: int = 24, seed: int = 0) -> FloatArray:
    """Limit cycle + small noise; λ ≈ 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + 0.02 * rng.standard_normal(n)


def synth_white(n: int = 4000, seed: int = 0) -> FloatArray:
    """iid noise: no attractor; divergence curve flat."""
    return np.random.default_rng(seed).standard_normal(n)


def bench_lyapunov(seed: int = 20261231 + 172) -> dict[str, float]:
    """SYNTHETIC benchmark: λ recovery on logistic/Hénon maps + FNN dims."""
    lam = largest_lyapunov(synth_logistic(seed=seed), dim=3, tau=1)
    lam_h = largest_lyapunov(synth_henon(seed=seed + 1), dim=3, tau=1)
    lam_p = largest_lyapunov(synth_periodic(seed=seed + 2), dim=3, tau=6)
    lam_w = largest_lyapunov(synth_white(seed=seed + 3), dim=3, tau=1)
    fnn = false_nearest_neighbors(synth_henon(seed=seed + 1), max_dim=5)
    cao = cao_dimension(synth_henon(seed=seed + 1), max_dim=5)
    cao_w = cao_dimension(synth_white(seed=seed + 3), max_dim=5)
    r1 = largest_lyapunov(synth_logistic(seed=seed), dim=3, tau=1)["lyap"]
    r2 = largest_lyapunov(synth_logistic(seed=seed), dim=3, tau=1)["lyap"]
    return {
        "synthetic_lyap_logistic": float(lam["lyap"]),
        "synthetic_lyap_logistic_err": abs(float(lam["lyap"]) - math.log(2.0)),
        "synthetic_lyap_henon": float(lam_h["lyap"]),
        "synthetic_lyap_henon_err": abs(float(lam_h["lyap"]) - 0.4192),
        "synthetic_lyap_periodic": float(lam_p["lyap"]),
        "synthetic_lyap_white": float(lam_w["lyap"]),
        "synthetic_lyap_chaos_ordering": float(lam["lyap"] > lam_p["lyap"] and lam["lyap"] > 0.5),
        "synthetic_fnn_dim2_frac": float(fnn[1]),
        "synthetic_fnn_dim1_frac": float(fnn[0]),
        "synthetic_cao_dim_henon": float(cao["dim_cao"]),
        "synthetic_cao_white_unsat": float(cao_w["dim_cao"] >= 5 or cao_w["dim_cao"] == -1),
        "synthetic_determinism": float(r1 == r2),
    }
