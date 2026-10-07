"""One-way MANOVA — omnibus multivariate group-mean tests (SYNTHETIC).

Anderson (1958), Rencher & Christensen (2012): for G groups observed
on p responses, the between-group SSP matrix H and within-group SSP
matrix E define four classical omnibus statistics through the
eigenvalues of E^{-1} H:

    Wilks   Lambda = prod 1/(1+lam_i)   (Bartlett chi^2 approx)
    Pillai  V      = sum  lam_i/(1+lam_i)
    Hotelling-Lawley T = sum lam_i
    Roy     theta  = max lam_i

Honesty: the bench plants a 3-group design shifted in 2 of 4
dimensions (Wilks p must reject) and a null design (p must respect
size ~0.05 nominal at a generous bound). The Bartlett chi^2
approximation is first-order — loose bounds documented. Fail-closed
on singleton groups, rank-deficient E, or non-finite data.

References: Wilks (1932); Pillai (1955); Bartlett (1938); Rencher &
Christensen (2012) "Methods of Multivariate Analysis" ch. 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def manova(groups: list[FloatArray]) -> dict[str, float]:
    """One-way MANOVA omnibus statistics for a list of group samples.

    Each element of ``groups`` is an (n_g, p) float array (n_g >= 3,
    p >= 2, at least 2 groups). Returns the four omnibus statistics
    and the Bartlett-chi^2 p-value for Wilks' Lambda.
    """
    if len(groups) < 2:
        raise ValueError("need >=2 groups")
    xs = [np.asarray(g, dtype=float) for g in groups]
    p = xs[0].shape[1] if xs[0].ndim == 2 else 0
    if any(
        g.ndim != 2 or g.shape[0] < 3 or g.shape[1] != p or not np.isfinite(g).all() for g in xs
    ):
        raise ValueError("bad group samples")
    if p < 2:
        raise ValueError("need >=2 responses")
    g_tot = len(xs)
    n = sum(g.shape[0] for g in xs)
    if n <= g_tot + p:
        raise ValueError("underdetermined: n <= G + p")
    gm = np.concatenate(xs, axis=0).mean(axis=0)
    h = np.zeros((p, p))
    e = np.zeros((p, p))
    for g in xs:
        d = g.mean(axis=0) - gm
        h += g.shape[0] * np.outer(d, d)
        c = g - g.mean(axis=0)
        e += c.T @ c
    lam = np.linalg.eigvalsh(np.linalg.solve(e, h))
    lam = np.clip(lam, 0.0, None)
    wilks = float(np.prod(1.0 / (1.0 + lam)))
    pillai = float(np.sum(lam / (1.0 + lam)))
    hotelling = float(lam.sum())
    roy = float(lam.max())
    # Bartlett (1938) approximation: -((n-1) - (p+G)/2) log Lambda ~ chi2[p(G-1)]
    df = p * (g_tot - 1)
    stat = -((n - 1) - (p + g_tot) / 2.0) * np.log(max(wilks, 1e-300))
    p_wilks = float(chi2.sf(max(0.0, stat), df))
    return {
        "wilks_lambda": wilks,
        "wilks_p": p_wilks,
        "pillai": pillai,
        "hotelling_lawley": hotelling,
        "roy_max": roy,
        "df": float(df),
    }


def bench_manova(seed: int = 20261231 + 416) -> dict[str, float]:
    """SYNTHETIC check — planted shift rejected, null respects size."""
    rng = np.random.default_rng(seed)
    shift = np.array([1.5, 1.2, 0.0, 0.0])
    groups = [rng.standard_normal((60, 4)) + s * shift for s in (0.0, 0.7, 1.3)]
    out = manova(groups)
    p_alt = out["wilks_p"]
    null = [rng.standard_normal((60, 4)) for _ in range(3)]
    p_null = manova(null)["wilks_p"]
    if p_alt > 0.01 or p_null < 0.02:
        raise ValueError(f"manova off: p_alt={p_alt:.4f} p_null={p_null:.4f}")
    return {
        "synthetic_manova_p_alt": p_alt,
        "synthetic_manova_p_null": p_null,
        "synthetic_manova_roy": out["roy_max"],
        "synthetic_score": 1.0,
    }
