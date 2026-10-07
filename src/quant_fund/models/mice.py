"""Multiple imputation by chained equations (MICE) with
predictive mean matching, plus Rubin's pooling rules.

van Buuren & Groothuis-Oudshoorn (2011): for a data matrix with
missing entries, iterate over the incomplete columns j:
fit a regression of x_j on the remaining (currently completed)
columns using observed rows, draw a posterior predictive value
for each missing row by predictive mean matching — pick the
donor among observed rows whose fitted value is nearest to the
drawn prediction (PMM keeps imputations inside the observed
support and robust to non-normality). Repeat for `n_iter`
sweeps; repeat the whole chain m times for m imputations.

Rubin (1987) pooling combines m per-imputation estimates:
q_bar = mean(q_m), U = mean(U_m), B = var(q_m), and

    T = U + (1 + 1/m) B,   se = sqrt(T),

with Barnard-Rubin (1999) adjusted degrees of freedom.

Honesty: linear-model PMM only (Gaussian columns), no
categorical handling — non-numeric columns fail closed at the
input check. Convergence is a fixed sweep count (documented);
the bench plants MAR missingness on one column where
complete-case mean is biased and the pooled imputed mean must
recover the population mean. Fail-closed on fully-missing
columns or no missing data.

References: van Buuren & Groothuis-Oudshoorn (2011) JSS 45:3;
Rubin (1987) "Multiple Imputation for Nonresponse in
Surveys"; Barnard & Rubin (1999) Biometrika 86:948; Little &
Rubin (2019) "Statistical Analysis with Missing Data".
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _pmm_fill(
    x: FloatArray,
    mask: BoolArray,
    miss_cols: list[int],
    rng: np.random.Generator,
    donors: int = 5,
) -> FloatArray:
    """One sweep of chained-equation PMM over miss_cols — the
    observed/donor pool is always the ORIGINALLY observed rows
    (van Buuren scheme), so mean-initialized draws do not enter
    the fit or the donor set."""
    n, p = x.shape
    out = x.copy()
    for j in miss_cols:
        obs = np.flatnonzero(~mask[:, j])
        mis = np.flatnonzero(mask[:, j])
        if mis.size == 0 or obs.size < 3:
            continue
        others = [c for c in range(p) if c != j]
        xo = out[np.ix_(obs, others)]
        xm = out[np.ix_(mis, others)]
        a = np.column_stack([np.ones(xo.shape[0]), xo])
        y_obs = out[obs, j]
        beta, *_ = np.linalg.lstsq(a, y_obs, rcond=None)
        fit_obs = a @ beta
        resid_sd = float(np.std(y_obs - fit_obs, ddof=1))
        am = np.column_stack([np.ones(xm.shape[0]), xm])
        pred = am @ beta + rng.normal(scale=resid_sd, size=mis.size)
        # donor matching: nearest observed fitted values
        for i, row in enumerate(mis):
            dist = np.abs(fit_obs - pred[i])
            k = int(min(donors, dist.size))
            pool = np.argpartition(dist, k - 1)[:k]
            donor = int(rng.choice(pool))
            out[row, j] = y_obs[donor]
    return out


def mice_impute(
    x: FloatArray,
    m: int = 5,
    n_iter: int = 10,
    donors: int = 5,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Return m completed datasets (PMM chained equations)."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 5 or xx.shape[1] < 2:
        raise ValueError("x must be n x p, n>=5, p>=2")
    mask = ~np.isfinite(xx)
    if not mask.any():
        raise ValueError("no missing entries")
    miss_cols = [j for j in range(xx.shape[1]) if mask[:, j].any()]
    for j in miss_cols:
        if mask[:, j].all():
            raise ValueError("fully missing column")
    datasets = []
    for rep in range(m):
        # init: column means
        cur = xx.copy()
        for j in miss_cols:
            obs = np.flatnonzero(np.isfinite(xx[:, j]))
            cur[mask[:, j], j] = xx[obs, j].mean()
        rr = np.random.default_rng(seed + rep * 7919)
        for _ in range(n_iter):
            cur = _pmm_fill(cur, mask, miss_cols, rr, donors=donors)
        datasets.append(cur)
    stacked = np.stack(datasets)
    return {
        "datasets": np.asarray(stacked, dtype=np.float64),
        "m": float(m),
        "n_missing": float(mask.sum()),
    }


def pool_estimates(
    estimates: FloatArray,
    ses: FloatArray,
) -> dict[str, float]:
    """Rubin's rules pooling for per-imputation point estimates."""
    q = np.asarray(estimates, dtype=np.float64)
    u = np.asarray(ses, dtype=np.float64)
    if q.shape != u.shape or q.size < 2:
        raise ValueError("need >=2 paired estimates")
    m = float(q.size)
    q_bar = float(q.mean())
    u_bar = float((u * u).mean())
    b = float(q.var(ddof=1))
    t_var = u_bar + (1.0 + 1.0 / m) * b
    se = math.sqrt(t_var)
    # Barnard-Rubin df
    lam = (1.0 + 1.0 / m) * b / max(t_var, 1e-12)
    df = max((m - 1.0) / max(lam * lam, 1e-12), 3.0)
    return {
        "q_bar": q_bar,
        "se_total": se,
        "between_var": b,
        "within_var": u_bar,
        "df": df,
        "fmi": float(lam),
    }


def bench_mice(seed: int = 20261231 + 455) -> dict[str, float]:
    """SYNTHETIC check — PMM imputation removes MAR bias."""
    rng = np.random.default_rng(seed)
    n = 600
    z = rng.normal(size=n)
    y = 2.0 + 1.5 * z + rng.normal(scale=0.5, size=n)
    # MAR: y missing more often when z large
    p_miss = 1.0 / (1.0 + np.exp(-(z - 0.5)))
    miss = rng.random(n) < p_miss * 0.6
    x = np.column_stack([z, y])
    x_m = x.copy()
    x_m[miss, 1] = np.nan
    out = mice_impute(x_m, m=5, n_iter=8, seed=seed)
    ds = np.asarray(out["datasets"], dtype=np.float64)
    ests = np.array([d[:, 1].mean() for d in ds])
    ses = np.array([d[:, 1].std(ddof=1) / np.sqrt(n) for d in ds])
    pooled = pool_estimates(ests, ses)
    q_bar = float(pooled["q_bar"])
    true_mean = float(y.mean())
    cc_mean = float(np.nanmean(x_m[:, 1]))
    err_imp = abs(q_bar - true_mean)
    err_cc = abs(cc_mean - true_mean)
    if err_imp >= err_cc or err_imp > 0.3:
        raise ValueError(f"mice off: imputed={q_bar:.3f} cc={cc_mean:.3f} true={true_mean:.3f}")
    return {
        "synthetic_imputed_err": err_imp,
        "synthetic_cc_err": err_cc,
        "synthetic_fmi": float(pooled["fmi"]),
        "synthetic_score": 1.0,
    }
