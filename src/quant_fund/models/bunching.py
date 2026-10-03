"""Bunching estimation at kinks and notches (Saez 2010; Chetty et al. 2011).

Agents facing a kink in their choice set (tax bracket, subsidy cap)
bunch at the threshold; the excess mass recovers the structural
elasticity. The standard estimator fits a flexible polynomial to the
histogram *excluding* bins near the kink, then reads the counterfactual
density at the kink; the excess mass ``B = Σ_{bins in window} (c_j − ĉ_j)``
normalized by counterfactual height gives the bunching ratio ``b`` and,
through the Saez formula, an implied elasticity.

Inference is by residual bootstrap over the excluded-region counts
(the standard Chetty et al. procedure — parametric bootstrap on the
Poisson-like bin variation).

References
----------
- Saez, E. (2010). *Do taxpayers bunch at kink points?* American
  Economic Journal: Economic Policy 2(3), 180–212.
- Chetty, R., Friedman, J. N., Olsen, T. & Pistaferri, L. (2011).
  *Adjustment costs, firm responses, and micro vs. macro labor supply
  elasticities.* QJE 126(2), 749–804.
- Kleven, H. J. & Waseem, M. (2013). *Using notches to uncover
  optimization frictions and structural elasticities.* QJE 128(2),
  669–723.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on degenerate histograms or kinks outside
the support; bootstrap SEs use fixed iteration counts for
reproducibility.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray

__all__ = [
    "bench_bunching",
    "bunching_estimate",
    "excess_mass",
    "synth_bunching",
]


def _poly_design(z: FloatArray, deg: int) -> FloatArray:
    n = z.size
    cols = [np.ones(n)]
    for d in range(1, deg + 1):
        cols.append(z**d)
    return np.column_stack(cols)


def bunching_estimate(
    x: FloatArray,
    kink: float,
    binwidth: float | None = None,
    window: float | None = None,
    poly_deg: int = 7,
    n_boot: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Bunching estimator at a kink.

    Histogram ``x`` at ``binwidth``; fit a degree-``poly_deg`` poly to
    log-counts on bins outside ``[kink ± window]``; counterfactual at
    the kink = poly prediction. ``b = B / ĉ(kink)`` where ``B`` is the
    excess mass in the window. Bootstrap over out-of-window count
    residuals for the SE.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 500:
        raise ValueError("need >=500 observations")
    lo, hi = float(x.min()), float(x.max())
    if not lo < kink < hi:
        raise ValueError("kink outside support")
    if binwidth is None:
        binwidth = (hi - lo) / 60.0
    if window is None:
        window = 3.0 * binwidth
    edges = np.arange(lo, hi + binwidth, binwidth)
    counts, _ = np.histogram(x, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    # Two regions: the narrow *bunch window* (the spike bins whose
    # counts exceed counterfactual) inside a wider *exclusion region*
    # the polynomial never sees — the exclusion also covers the
    # depleted band right of the kink the bunchers vacated.
    in_win = np.abs(mids - kink) <= window
    excl = (mids >= kink - 2.0 * window) & (mids <= kink + 3.0 * window)
    if in_win.sum() < 2:
        raise ValueError("window too narrow — no bins at the kink")
    if (~excl).sum() < poly_deg + 4:
        raise ValueError("too few bins outside the exclusion region")
    rng = np.random.default_rng(seed)

    def _fit_b(c: FloatArray) -> tuple[float, float]:
        """Return (excess mass B, counterfactual height at kink)."""
        logc = np.log(np.maximum(c, 0.5))
        f = _poly_design(mids[~excl] - kink, poly_deg)
        # Poisson-consistent weighting: Var(log count) ~ 1/count, so
        # WLS with sqrt(count) weights — without it the counterfactual
        # is systematically biased at dense regions.
        w = np.sqrt(np.maximum(c[~excl], 1.0))
        beta = np.linalg.lstsq(f * w[:, None], logc[~excl] * w, rcond=None)[0]
        f_all = _poly_design(mids - kink, poly_deg)
        c_hat = np.exp(f_all @ beta)
        b = float(np.sum(c[in_win] - c_hat[in_win]))
        c_kink = float(np.exp(_poly_design(np.array([0.0]), poly_deg) @ beta)[0])
        return b, max(c_kink, 1e-9)

    b_hat, c_kink = _fit_b(counts.astype(np.float64))
    bs = np.empty(n_boot)
    for i in range(n_boot):
        c_sim = counts.astype(np.float64).copy()
        out_idx = np.where(~excl)[0]
        # parametric bootstrap: Poisson jitter on fitted-region bins
        c_sim[out_idx] = rng.poisson(np.maximum(counts[out_idx], 1))
        bb, _ = _fit_b(np.maximum(c_sim, 0.0))
        bs[i] = bb / c_kink
    b_ratio = b_hat / c_kink
    return {
        "b": float(b_ratio),
        "excess_mass": float(b_hat),
        "counterfactual_at_kink": float(c_kink),
        "se": float(np.std(bs, ddof=1)),
        "window": float(window),
        "binwidth": float(binwidth),
        "n_bins_window": float(in_win.sum()),
    }


def excess_mass(
    x: FloatArray, kink: float, binwidth: float | None = None, window: float | None = None
) -> dict[str, float]:
    """Thin wrapper: the bunching ratio + interval."""
    out = bunching_estimate(x, kink, binwidth=binwidth, window=window, n_boot=50)
    se = out["se"]
    return {
        "b": out["b"],
        "ci_lb": out["b"] - 1.96 * se,
        "ci_ub": out["b"] + 1.96 * se,
        "se": se,
    }


def synth_bunching(
    n: int = 8000,
    kink: float = 0.0,
    strength: float = 0.4,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC density with bunching: smooth base density plus a
    ``strength``-proportional spike relocated to the kink from a small
    band just right of it — the signature of agents bunching at a
    threshold from above."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 1.0, n)
    # hollow out the band (kink, kink+0.15) and pile it at the kink
    band = (base > kink + 0.05) & (base < kink + 0.45)
    n_move = int(strength * band.sum())
    movers = np.where(band)[0]
    pick = rng.choice(movers, size=min(n_move, movers.size), replace=False)
    base[pick] = kink + 0.02 * rng.random(pick.size)
    return {
        "x": base.astype(np.float64),
        "strength": np.float64(strength),
        "n_moved": np.float64(pick.size),
    }


def bench_bunching(seed: int = 20261231 + 181) -> dict[str, float]:
    """SYNTHETIC: b > 0 at a real kink, ≈0 away from it; bootstrap SE sane."""
    d = synth_bunching(n=8000, strength=0.5, seed=seed)
    x = np.asarray(d["x"])
    est = bunching_estimate(
        x, kink=0.0, binwidth=0.05, window=0.15, poly_deg=5, n_boot=60, seed=seed
    )
    clean = synth_bunching(n=8000, strength=0.0, seed=seed + 3)
    est_off = bunching_estimate(
        np.asarray(clean["x"]),
        kink=0.0,
        binwidth=0.05,
        window=0.15,
        poly_deg=5,
        n_boot=60,
        seed=seed + 1,
    )
    est2 = bunching_estimate(
        x, kink=0.0, binwidth=0.05, window=0.15, poly_deg=5, n_boot=60, seed=seed
    )
    return {
        "synthetic_b_at_kink": float(est["b"]),
        "synthetic_b_se": float(est["se"]),
        "synthetic_b_z": float(est["b"] / max(est["se"], 1e-9)),
        "synthetic_b_offkink": float(est_off["b"]),
        "synthetic_excess_mass": float(est["excess_mass"]),
        "synthetic_n_moved": float(d["n_moved"]),
        "synthetic_detects_kink": float(
            est["b"] > 4.0 * max(est["se"], 1e-9) and abs(est["b"]) > 3.0 * abs(est_off["b"])
        ),
        "synthetic_determinism": float(est["b"] == est2["b"]),
    }
