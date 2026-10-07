"""Rosenbaum sensitivity analysis for observational matched-pair studies (SYNTHETIC).

Even when matching balances observed covariates, a hidden confounder
can tilt the odds of treatment by a factor Γ > 1 within matched
pairs. Rosenbaum's Γ-bounds propagate that worst-case tilt through
the Wilcoxon signed-rank statistic: at each Γ, treated-minus-control
pair differences get weights U_i in [1/(1+Γ), Γ/(1+Γ)] and the
test statistic T_G follows an approximate normal with expectation
and variance obtained by summing the bounds over pairs. The
critical Γ* at which the upper-bound p-value crosses α is the
study's sensitivity to hidden bias — larger Γ*, more robust.

Honesty: synthetic matched pairs are generated with a known
treatment effect and a planted unobserved confounder of known
strength; the bench checks the recovered Γ* against the truth and
that Γ* shrinks with stronger confounding — a proper diagnostic,
never market evidence.

References:
- Rosenbaum, P. R. (1987). Sensitivity analysis for certain
  permutation inferences in matched observational studies.
  *Biometrika* 74 — the Γ model.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.),
  Springer — Wilcoxon signed-rank bounds, §4.
- Rosenbaum, P. R. (2005). Sensitivity analysis in
  observational studies. In *Encyclopedia of Statistics in
  Behavioral Science* — critical-Γ interpretation.
- DiPrete, T. A., Gangl, M. (2004). Assessing bias in the
  estimation of causal effects. *Sociological Methodology* 34 —
  applied Γ-threshold practice.

Composition: numpy only — Wilcoxon signed-rank bounds over
matched pairs; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

FloatArray = NDArray[np.float64]


def _pair_stats(d: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Wilcoxon signed-rank decomposition: abs-diff ranks a_i and
    positive-difference indicators c_i (zeros dropped)."""
    ad = np.abs(d)
    keep = ad > 0
    a = rankdata(ad[keep])
    c = (d[keep] > 0).astype(np.float64)
    return a, c


def _pvalue_bound(d: FloatArray, gamma: float, upper: bool) -> float:
    """Bound p-value of the Wilcoxon signed-rank statistic at
    sensitivity level Γ (upper=True → worst case for a
    positive-effect alternative, i.e. all pair weights at
    Γ/(1+Γ))."""
    a, c = _pair_stats(d)
    if a.size == 0:
        raise ValueError("no nonzero pair differences")
    p_hi = gamma / (1.0 + gamma)
    p_lo = 1.0 / (1.0 + gamma)
    n = a.size
    t_obs = float(np.sum(a * c))
    if upper:
        mu = n * p_hi * float(a.mean())
        var = n * p_hi * (1.0 - p_hi) * float(np.sum(a**2) / n)
    else:
        mu = n * p_lo * float(a.mean())
        var = n * p_lo * (1.0 - p_lo) * float(np.sum(a**2) / n)
    if var <= 0:
        raise ValueError("degenerate rank variance")
    return float(1.0 - norm.cdf((t_obs - mu) / np.sqrt(var)))


def sensitivity_bounds(
    diff: FloatArray,
    gammas: FloatArray | None = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Rosenbaum bounds: for each Γ in ``gammas`` the upper- and
    lower-bound p-value of the signed-rank test, plus the
    critical Γ* where the upper-bound p-value first exceeds
    ``alpha`` (NaN-safe: grid max if it never crosses, floor
    Γ=1 if already insignificant). ``diff`` = treated −
    control within each matched pair."""
    d = np.asarray(diff, dtype=np.float64)
    if d.size < 10 or d.ndim != 1:
        raise ValueError("1-D matched-pair differences, n>=10 required")
    if not np.all(np.isfinite(d)):
        raise ValueError("finite differences required")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha in (0, 0.5) required")
    g = np.asarray(gammas, dtype=np.float64) if gammas is not None else np.linspace(1.0, 4.0, 31)
    if g.ndim != 1 or g.size < 4 or np.any(g < 1.0):
        raise ValueError("increasing gammas >= 1 required")
    p_up = np.array([_pvalue_bound(d, float(gg), True) for gg in g])
    p_lo = np.array([_pvalue_bound(d, float(gg), False) for gg in g])
    crossed = np.nonzero(p_up > alpha)[0]
    gamma_star = float(g[crossed[0]]) if crossed.size else float(g[-1])
    return {
        "gamma_star": gamma_star,
        "p_at_gamma1": float(p_up[0]),
        "p_upper_max": float(p_up[-1]),
        "p_lower_max": float(p_lo[-1]),
        "n_pairs": float(np.sum(np.abs(d) > 0)),
        "wilcoxon_t": float(np.sum(rankdata(np.abs(d)[np.abs(d) > 0]) * (d[np.abs(d) > 0] > 0))),
    }


def synth_matched_pairs(
    n: int = 200,
    effect: float = 0.8,
    confound: float = 0.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Matched pair differences: d_i = effect + u_i + e_i where
    u_i ~ N(0, confound²) is a hidden tilt (≈ log-odds odds
    ratio ≈ exp(|u|); report Γ ≈ exp(sd) as the design truth)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, confound, n) if confound > 0 else np.zeros(n)
    e = rng.normal(0.0, 1.0, n)
    d = effect + u + e
    return {"diff": np.asarray(d, dtype=np.float64), "u": u}


def bench_rosenbaum_sensitivity(seed: int = 20261231 + 269) -> dict[str, float]:
    """Sensitivity self-check: with a clear effect and no
    confounding, Γ* should be well above 1 (robust); with
    confounding the same Γ* shrinks. All ``synthetic_*``."""
    d_clean = synth_matched_pairs(effect=0.8, confound=0.0, seed=seed)
    d_conf = synth_matched_pairs(effect=0.8, confound=0.7, seed=seed)
    clean = sensitivity_bounds(np.asarray(d_clean["diff"]))
    conf = sensitivity_bounds(np.asarray(d_conf["diff"]))
    out2 = sensitivity_bounds(np.asarray(d_conf["diff"]))
    return {
        "synthetic_gamma_star_clean": clean["gamma_star"],
        "synthetic_gamma_star_confounded": conf["gamma_star"],
        "synthetic_p_gamma1": clean["p_at_gamma1"],
        "synthetic_detects": float(
            clean["gamma_star"] > 1.5
            and conf["gamma_star"] < clean["gamma_star"]
            and clean["p_at_gamma1"] < 0.01
        ),
        "synthetic_determinism": float(out2["gamma_star"] == conf["gamma_star"]),
    }
