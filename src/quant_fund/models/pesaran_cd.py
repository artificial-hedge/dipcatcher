"""Pesaran (2004) CD test for cross-sectional dependence.

After a panel regression, errors should be independent across
units if the model captures the common structure. The CD
statistic

  CD = sqrt(2T / (N(N−1))) Σ_{i<j} ρ̂_ij,  ρ̂_ij = corr(ê_i·, ê_j·)

is asymptotically N(0,1) under cross-sectional independence for
N, T large in either order — robust to heterogeneous dynamics
where Breusch-Pagan's LM degenerates. The companion average
pairwise correlation ρ̄ reports the magnitude of residual
co-movement.

Honesty: synthetic panels with a known common factor strength
λ; the bench checks the CD z-score grows with λ and stays near
zero under independence — a proper diagnostic, never market
evidence.

References:
- Pesaran, M. H. (2004). General diagnostic tests for cross
  section dependence in panels. *CESifo Working Paper* 1229 —
  the CD statistic and its N(0,1) limit.
- Pesaran, M. H. (2015). Testing weak cross-sectional
  dependence in large panels. *Econometric Reviews* 34 —
  the weak-dependence power envelope.
- Hsiao, C., Pesaran, M. H., Pick, A. (2012). Diagnostic
  tests of cross-sectional independence for limited dependent
  variable panel data models. *Oxford Bulletin* 74 — extent
  of the independence null.
- Breusch, T., Pagan, A. (1980). The Lagrange multiplier test
  and its applications to model specification. *Review of
  Economic Studies* 47 — the LM the CD test replaces.

Composition: numpy + scipy only — pairwise residual
correlations and the CD z-score; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def pesaran_cd(resid: FloatArray) -> dict[str, float]:
    """CD test on a residual matrix ``resid`` shaped (T, N) —
    columns are unit time series. Returns the z-statistic,
    two-sided p-value, and the average pairwise correlation."""
    r = np.asarray(resid, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 10 or r.shape[1] < 5:
        raise ValueError("(T,N) matrix with T>=10, N>=5 required")
    if not np.all(np.isfinite(r)):
        raise ValueError("finite residuals required")
    t, n = r.shape
    sd = r.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("degenerate unit with zero variance")
    c = np.asarray(np.corrcoef(r, rowvar=False), dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    rhos = c[iu]
    m = n * (n - 1) // 2
    cd = float(np.sqrt(2.0 * t / (n * (n - 1))) * np.sum(rhos))
    return {
        "cd": cd,
        "p": float(2.0 * norm.sf(abs(cd))),
        "rho_bar": float(np.mean(rhos)),
        "rho_bar_abs": float(np.mean(np.abs(rhos))),
        "n_pairs": float(m),
    }


def synth_cd_panel(
    t: int = 60,
    n: int = 40,
    lam: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Panel residuals: e_it = λ_i·f_t + u_it — factor strength
    λ drives cross-sectional correlation ≈ λ²/(1+λ²)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 1.0, t)
    e = lam * f[:, None] + rng.normal(0.0, 1.0, (t, n))
    return {"resid": np.asarray(e, dtype=np.float64)}


def bench_pesaran_cd(seed: int = 20261231 + 272) -> dict[str, float]:
    """CD self-check: strong factor (λ=.7) → large positive CD
    and ρ̄≈λ²/(1+λ²)≈.33; independent residuals → CD≈0.
    All ``synthetic_*``."""
    dep = pesaran_cd(np.asarray(synth_cd_panel(lam=0.7, seed=seed)["resid"]))
    indep = pesaran_cd(np.asarray(synth_cd_panel(lam=0.0, seed=seed)["resid"]))
    out2 = pesaran_cd(np.asarray(synth_cd_panel(lam=0.7, seed=seed)["resid"]))
    return {
        "synthetic_cd_dep": dep["cd"],
        "synthetic_cd_indep": indep["cd"],
        "synthetic_rho_bar_dep": dep["rho_bar"],
        "synthetic_rho_bar_theory": 0.7**2 / (1.0 + 0.7**2),
        "synthetic_p_dep": dep["p"],
        "synthetic_p_indep": indep["p"],
        "synthetic_detects": float(
            dep["p"] < 0.001 and indep["p"] > 0.05 and 0.15 < dep["rho_bar"] < 0.55
        ),
        "synthetic_determinism": float(out2["cd"] == dep["cd"]),
    }
