"""GSADF recursive bubble detection (Phillips-Shi-Yu).

References
----------
- Phillips, P.C.B., Shi, S. & Yu, J. (2015). "Testing for Multiple
  Bubbles 1: Historical Episodes of Exuberance and Collapse in the
  S&P 500." *International Economic Review* 56(4), 1043-1078.
- Phillips, P.C.B., Shi, S. & Yu, J. (2015). "Testing for Multiple
  Bubbles 2: Limit Theory of Real-Time Detectors." *International
  Economic Review* 56(4), 1079-1134.
- Phillips, P.C.B., Wu, Y. & Yu, J. (2011). "Explosive Behavior in the
  1990s Nasdaq: When Did Exuberance Escalate Asset Values?"
  *International Economic Review* 52(1), 201-226.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Right-tailed ADF regression ``d y_t = a + b y_{t-1} + sum_k c_k d
y_{t-k} + e`` estimated by least squares on expanding and rolling
sub-windows ``[r1, r2]`` of the series. ``bsadf[r2]`` is the sup over
``r1 <= r2 - r0`` of the t-statistic on ``b``; ``gsadf`` is the sup
over all (r1, r2); ``sadf`` sups over r1 = 0 only. Date-stamping
flags intervals where ``bsadf`` exceeds a (1 - alpha) critical value;
critical values come from a deterministic Monte Carlo under the
random-walk null (``gsadf_critical_values``). The synth plants one
mildly explosive episode inside a random walk — detection must
overlap the planted window without flagging the tranquil prefix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def adf_tstat(y: FloatArray, lags: int = 0) -> float:
    """Right-tailed augmented Dickey-Fuller t-statistic on the level."""
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 1 or yy.size < 6:
        raise ValueError("series too short")
    if not np.all(np.isfinite(yy)):
        raise ValueError("non-finite series")
    if lags < 0 or lags > yy.size // 4:
        raise ValueError("bad lag order")
    if lags == 0:
        dy = np.diff(yy)
        stats = _adf0_stats(yy[:-1], dy)
        return float(stats[0, 0])
    dy = np.diff(yy)
    n = dy.size - lags
    if n < 4:
        raise ValueError("too few usable differences")
    ylag = yy[lags:-1][:n]
    cols = [np.ones(n), ylag]
    for k in range(1, lags + 1):
        cols.append(dy[lags - k : dy.size - k])
    dep = dy[lags:]
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, dep, rcond=None)
    resid = dep - x @ beta
    dof = max(n - x.shape[1], 1)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(max(s2 * xtx_inv[1, 1], 1e-18)))
    return float(beta[1] / se)


def _adf0_stats(
    x_arr: FloatArray,
    z_arr: FloatArray,
) -> FloatArray:
    """Vectorized no-augmentation ADF t-stats for pairs of arrays.

    ``x_arr`` (levels) and ``z_arr`` (first differences) must be
    aligned so the design is ``z_t = a + b x_t + e`` row-wise. For a
    single window pass slices of matching length; for the GSADF scan
    pass the arrays of cumulative window moments instead (see
    ``_window_tstat``).
    """
    xx = np.asarray(x_arr, dtype=np.float64)
    zz = np.asarray(z_arr, dtype=np.float64)
    n = xx.size
    if n < 4:
        raise ValueError("too few observations")
    xm = np.mean(xx)
    zm = np.mean(zz)
    xc = xx - xm
    zc = zz - zm
    sxx = float(np.sum(xc * xc))
    szz = float(np.sum(zc * zc))
    sxz = float(np.sum(xc * zc))
    if sxx <= 0.0:
        raise ValueError("degenerate regressor")
    b = sxz / sxx
    rss = szz - sxz * sxz / sxx
    s2f: float = (rss if rss > 1e-18 else 1e-18) / (n - 2)
    se = float(np.sqrt(s2f / sxx))
    return np.asarray([[b / se]])


def _window_tstat(
    sy: FloatArray,
    sz: FloatArray,
    syy: FloatArray,
    szz: FloatArray,
    syz: FloatArray,
    lo: int,
    hi: int,
) -> float:
    """O(1) no-augmentation ADF t-stat over the index window [lo, hi).

    The five arguments are prefix-sum arrays of ``y_lag``, ``dy``,
    ``y_lag^2``, ``dy^2`` and ``y_lag*dy`` (index 0 = empty window),
    so window sums are prefix differences.
    """
    n = hi - lo
    if n < 4:
        return -np.inf
    sy_ = float(sy[hi] - sy[lo])
    sz_ = float(sz[hi] - sz[lo])
    syy_ = float(syy[hi] - syy[lo])
    szz_ = float(szz[hi] - szz[lo])
    syz_ = float(syz[hi] - syz[lo])
    xm = sy_ / n
    zm = sz_ / n
    sxx = syy_ - n * xm * xm
    szz_c = szz_ - n * zm * zm
    sxz = syz_ - n * xm * zm
    if sxx <= 0.0:
        return -np.inf
    b = sxz / sxx
    rss = szz_c - sxz * sxz / sxx
    s2f: float = (rss if rss > 1e-18 else 1e-18) / (n - 2)
    se = float(np.sqrt(s2f / sxx))
    return float(b / se)


def gsadf(
    y: FloatArray,
    r0: float | None = None,
    lags: int = 0,
) -> dict[str, float | FloatArray]:
    """Generalized sup-ADF scan over sub-windows of the series.

    ``r0`` is the minimal window fraction (default ``0.01 + 1.8 /
    sqrt(T)`` per PSY). Returns the sup statistics and the
    backward-recursive ``bsadf`` / ``sadf`` sequences (NaN before the
    first feasible end point).
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 1 or not np.all(np.isfinite(yy)):
        raise ValueError("series must be finite 1-D")
    t = yy.size
    if t < 20:
        raise ValueError("series too short")
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(t)
    if not (0.05 <= r0 <= 0.9):
        raise ValueError("r0 out of range")
    w = max(int(np.floor(r0 * t)), lags + 8)
    bsadf = np.full(t, np.nan)
    sadf = np.full(t, np.nan)
    if lags == 0:
        dy = np.diff(yy)
        yl = yy[:-1]
        sy = np.concatenate([[0.0], np.cumsum(yl)])
        sz = np.concatenate([[0.0], np.cumsum(dy)])
        syy = np.concatenate([[0.0], np.cumsum(yl * yl)])
        szz = np.concatenate([[0.0], np.cumsum(dy * dy)])
        syz = np.concatenate([[0.0], np.cumsum(yl * dy)])
        # subseries y[r1:r2] (length >= w) uses regression rows
        # j in [r1, r2 - 1): pairs (y[j], dy[j]) for j < r2 - 1.
        for r2 in range(w + 1, t + 1):
            starts = np.arange(0, r2 - w + 1)
            hi = r2 - 1
            nw = (hi - starts).astype(np.float64)
            syw = sy[hi] - sy[starts]
            szw = sz[hi] - sz[starts]
            syyw = syy[hi] - syy[starts]
            szzw = szz[hi] - szz[starts]
            syzw = syz[hi] - syz[starts]
            xm = syw / nw
            zm = szw / nw
            sxx = syyw - nw * xm * xm
            sxz = syzw - nw * xm * zm
            szzv = szzw - nw * zm * zm
            valid = sxx > 0.0
            sxx_s = np.where(valid, sxx, 1.0)
            b = np.where(valid, sxz / sxx_s, -np.inf)
            rss = szzv - np.where(valid, sxz * sxz / sxx_s, 0.0)
            s2 = np.maximum(rss, 1e-18) / np.maximum(nw - 2.0, 1.0)
            tstat = np.where(valid, b / np.sqrt(s2 / sxx_s), -np.inf)
            tstat = np.where(nw >= 4.0, tstat, -np.inf)
            bsadf[r2 - 1] = float(np.max(tstat))
            sadf[r2 - 1] = _window_tstat(sy, sz, syy, szz, syz, 0, r2 - 1)
    else:
        for r2 in range(w, t + 1):
            best = -np.inf
            for r1 in range(0, r2 - w + 1):
                s = adf_tstat(yy[r1:r2], lags=lags)
                if s > best:
                    best = s
            bsadf[r2 - 1] = best
        for r2 in range(w, t + 1):
            sadf[r2 - 1] = adf_tstat(yy[:r2], lags=lags)
    finite = bsadf[np.isfinite(bsadf)]
    finite_s = sadf[np.isfinite(sadf)]
    if finite.size == 0 or finite_s.size == 0:
        raise ValueError("no feasible window")
    return {
        "gsadf": float(np.max(finite)),
        "sadf": float(np.max(finite_s)),
        "bsadf": bsadf,
        "sadf_seq": sadf,
        "r0": float(r0),
        "min_window": float(w),
    }


def date_stamp(
    bsadf: FloatArray,
    cv: float,
    min_len: int = 1,
) -> FloatArray:
    """Indicator series of explosive episodes (bsadf above ``cv``).

    ``min_len`` merges adjacent flagged points separated by a gap of
    one observation (PSY consolidation rule) and drops episodes
    shorter than ``min_len`` samples.
    """
    bb = np.asarray(bsadf, dtype=np.float64)
    if bb.ndim != 1 or not np.isfinite(cv):
        raise ValueError("bad inputs")
    flags = (bb > cv).astype(np.float64)
    flags[np.isnan(bb)] = 0.0
    if min_len > 1:
        # PSY consolidation: bridge flagged runs separated by exactly one
        # unflagged observation before the length filter applies.
        idx0 = np.flatnonzero(flags)
        if idx0.size >= 2:
            gaps = np.where(np.diff(idx0) == 2)[0]
            flags[idx0[gaps] + 1] = 1.0
        idx = np.flatnonzero(flags)
        if idx.size:
            keep = np.zeros_like(flags)
            start = idx[0]
            prev = idx[0]
            for i in idx[1:]:
                if i - prev > 1:
                    if prev - start + 1 >= min_len:
                        keep[start : prev + 1] = 1.0
                    start = i
                prev = i
            if prev - start + 1 >= min_len:
                keep[start : prev + 1] = 1.0
            flags = keep
    return flags


def gsadf_critical_values(
    t: int,
    r0: float | None = None,
    lags: int = 0,
    n_rep: int = 500,
    seed: int = 20261231,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Monte-Carlo critical values under the driftless RW null.

    Deterministic given ``seed``: replicates unit-root Gaussian
    random walks of length ``t`` and collects the GSADF / SADF sup
    statistics, returning the requested quantiles keyed as
    ``gsadf_q95`` / ``sadf_q95`` etc.
    """
    if t < 20 or n_rep < 50:
        raise ValueError("t/n_rep too small")
    rng = np.random.default_rng(seed)
    g = np.empty(n_rep)
    s = np.empty(n_rep)
    for i in range(n_rep):
        rw = np.cumsum(rng.standard_normal(t))
        scan = gsadf(rw, r0=r0, lags=lags)
        g[i] = float(scan["gsadf"])
        s[i] = float(scan["sadf"])
    res: dict[str, float] = {}
    for q in quantiles:
        tag = int(round(q * 100))
        res[f"gsadf_q{tag}"] = float(np.quantile(g, q))
        res[f"sadf_q{tag}"] = float(np.quantile(s, q))
    return res


def synth_gsadf(
    seed: int = 20261231 + 294,
    t: int = 400,
    start: int = 200,
    length: int = 60,
    rho: float = 1.04,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC random walk with one planted explosive episode.

    A driftless unit root for ``t`` samples; inside
    ``[start, start + length)`` the AR coefficient lifts to ``rho``
    and an extra positive drift is added, after which the level
    partially collapses — the canonical PSY bubble shape.
    """
    rng = np.random.default_rng(seed)
    if not (0 < start < t - length):
        raise ValueError("episode out of range")
    y = np.empty(t)
    y[0] = 100.0
    for i in range(1, t):
        eps = rng.standard_normal()
        if start <= i < start + length:
            y[i] = rho * y[i - 1] + 0.15 + eps
        elif i == start + length:
            y[i] = 0.75 * y[i - 1] + eps
        else:
            y[i] = y[i - 1] + eps
    episode = np.zeros(t)
    episode[start : start + length] = 1.0
    return {
        "y": y,
        "episode": episode,
        "start": float(start),
        "end": float(start + length),
    }


def bench_gsadf(seed: int = 20261231 + 294) -> dict[str, float]:
    """Wave-51 self-check: the planted bubble is date-stamped."""
    d = synth_gsadf(seed=seed)
    y = np.asarray(d["y"])
    out = gsadf(y)
    bsadf = np.asarray(out["bsadf"])
    cvs = gsadf_critical_values(y.size, n_rep=120, seed=seed + 1)
    flags = date_stamp(bsadf, cvs["gsadf_q95"], min_len=3)
    episode = np.asarray(d["episode"])
    overlap = float(np.sum(flags * episode))
    flagged = float(np.sum(flags))
    tranquil_fp = float(np.sum(flags[: int(d["start"]) - 10]))
    ok = overlap >= 15.0 and flagged > 0.0 and tranquil_fp <= 2.0
    return {
        "synthetic_gsadf": float(out["gsadf"]),
        "synthetic_cv95": cvs["gsadf_q95"],
        "synthetic_overlap": overlap,
        "synthetic_flagged": flagged,
        "synthetic_tranquil_fp": tranquil_fp,
        "synthetic_score": float(ok),
    }
