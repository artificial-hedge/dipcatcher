"""Self-exciting threshold autoregression (SETAR) (SYNTHETIC).

The series switches AR regimes on its own lagged level:
``y_t = φ1·y_{t-1} + ε`` when ``y_{t-d} ≤ c``, else
``φ2·y_{t-1} + ε``. The threshold c is estimated by conditional
least squares over a grid; regime-specific dynamics capture
asymmetric adjustment (fast mean reversion in one regime,
persistence in the other) that a linear AR cannot.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure regime detection on generated
two-regime series — never market evidence.

References:
- Tong, H. (1978). On a threshold model. In *Pattern Recognition
  and Signal Processing* (ed. C. H. Chen) — the SETAR model.
- Tong, H., Lim, K. S. (1980). Threshold autoregression, limit
  cycles and cyclical data. *JRSS-B* 42, 245-292 — CLS estimation
  and the threshold grid.
- Hansen, B. E. (1999). Testing for linearity. *Journal of
  Economic Surveys* 13, 551-576 — the sup-F linearity test
  implemented here (bootstrap p-value).
- Enders, W., Granger, C. W. J. (1998). Unit-root tests and
  asymmetric adjustment with TAR and M-TAR models. *JASA* 93.

Composition: pure numpy — grid-searched threshold, per-regime OLS,
recursive-residual sup-F test; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _fit_regime(y: FloatArray, x: FloatArray) -> tuple[FloatArray, float]:
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    r = y - x @ b
    return b, float(r @ r / max(y.size - x.shape[1], 1))


def setar_fit(
    y: FloatArray,
    delay: int = 1,
    trim: float = 0.15,
) -> dict[str, float]:
    """Two-regime SETAR(1) with grid-searched threshold.

    Regime switch on ``y_{t-delay}``; threshold grid over the
    central (trim, 1-trim) quantile band."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = yy.size
    if n < 60:
        raise ValueError("need >=60 observations")
    if not np.all(np.isfinite(yy)):
        raise ValueError("finite series required")
    if delay < 1 or delay > 4:
        raise ValueError("delay in 1..4")
    if not 0.05 < trim < 0.45:
        raise ValueError("trim in (0.05, 0.45)")

    dep = yy[1 + delay - 1 :]
    lag = yy[delay - 1 : n - 1]  # y_{t-1}
    swt = yy[: n - delay]  # switching variable y_{t-delay}
    # align: rows i ↔ t = delay + i (for i in 0..n-1-delay)
    # dep[i] = y[delay + i], lag[i] = y[delay - 1 + i], swt[i] = y[i]
    if dep.size != lag.size or dep.size != swt.size:
        dep = yy[delay:]
        lag = yy[delay - 1 : n - 1]
        swt = yy[: n - delay]
        m = min(dep.size, lag.size, swt.size)
        dep, lag, swt = dep[:m], lag[:m], swt[:m]

    qs = np.quantile(swt, np.linspace(trim, 1 - trim, 25))
    best = None
    for c in np.unique(qs):
        lo = swt <= c
        if lo.sum() < 15 or (~lo).sum() < 15:
            continue
        xlo = np.column_stack([np.ones(int(lo.sum())), lag[lo]])
        xhi = np.column_stack([np.ones(int((~lo).sum())), lag[~lo]])
        _, s0 = _fit_regime(dep[lo], xlo)
        _, s1 = _fit_regime(dep[~lo], xhi)
        sse = s0 * max(int(lo.sum()) - 2, 1) + s1 * max(int((~lo).sum()) - 2, 1)
        if best is None or sse < best[0]:
            best = (sse, float(c), lo)
    if best is None:
        raise ValueError("no feasible threshold split")
    sse_t, c_hat, lo = best

    xlo = np.column_stack([np.ones(int(lo.sum())), lag[lo]])
    xhi = np.column_stack([np.ones(int((~lo).sum())), lag[~lo]])
    b_lo, s2_lo = _fit_regime(dep[lo], xlo)
    b_hi, s2_hi = _fit_regime(dep[~lo], xhi)

    # linear AR(1) reference
    xlin = np.column_stack([np.ones(lag.size), lag])
    b_lin, s2_lin = _fit_regime(dep, xlin)
    sse_lin = s2_lin * (lag.size - 2)
    # sup-F: (SSE_lin - SSE_tar)/SSE_tar * df
    f_stat = (sse_lin - sse_t) / max(sse_t, 1e-12) * (lag.size - 4)
    f_stat = float(max(f_stat, 0.0))

    return {
        "c_hat": c_hat,
        "phi_lo": float(b_lo[1]),
        "phi_hi": float(b_hi[1]),
        "const_lo": float(b_lo[0]),
        "const_hi": float(b_hi[0]),
        "phi_linear": float(b_lin[1]),
        "sse_tar": float(sse_t),
        "sse_linear": float(sse_lin),
        "f_linearity": f_stat,
        "share_lo": float(np.mean(lo)),
        "n": float(n),
    }


def setar_linearity_p(
    y: FloatArray,
    delay: int = 1,
    n_boot: int = 60,
    seed: int = 0,
) -> float:
    """Hansen sup-F bootstrap p-value for SETAR-vs-linear."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    f_obs = setar_fit(yy, delay=delay)["f_linearity"]
    rng = np.random.default_rng(seed)
    # linear AR under null → bootstrap series
    lag = yy[delay - 1 : -1]
    dep = yy[delay:]
    b = np.linalg.lstsq(np.column_stack([np.ones(lag.size), lag]), dep, rcond=None)[0]
    resid = dep - b[0] - b[1] * lag
    resid = resid - resid.mean()
    sd = float(resid.std())
    cnt = 0
    for _ in range(n_boot):
        sim = np.zeros(yy.size)
        sim[0] = yy[0]
        e = rng.normal(0.0, sd, yy.size)
        for t in range(1, yy.size):
            sim[t] = b[0] + b[1] * sim[t - 1] + e[t]
        try:
            f_sim = setar_fit(sim, delay=delay)["f_linearity"]
        except ValueError:
            continue
        cnt += f_sim >= f_obs
    return float((1 + cnt) / (1 + n_boot))


def synth_setar(
    n: int = 600,
    threshold: float = 0.5,
    phi_lo: float = 0.85,
    phi_hi: float = 0.3,
    seed: int = 0,
) -> FloatArray:
    """SETAR DGP: persistent low regime, mean-reverting high regime."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        if y[t - 1] <= threshold:
            y[t] = 0.05 + phi_lo * y[t - 1] + rng.normal(0.0, 0.15)
        else:
            y[t] = 0.4 + phi_hi * y[t - 1] + rng.normal(0.0, 0.15)
    return y


def bench_threshold_ar(seed: int = 20261231 + 218) -> dict[str, float]:
    """SETAR self-check: threshold + regime coefficients recovered;
    linear AR misses the asymmetry. All ``synthetic_*``."""
    y = synth_setar(seed=seed)
    out = setar_fit(y)
    p_val = setar_linearity_p(y, seed=seed)
    # true null: single-regime AR(1) — no regime structure at all
    rng0 = np.random.default_rng(seed + 1)
    y0 = np.zeros(600)
    for t in range(1, 600):
        y0[t] = 0.1 + 0.6 * y0[t - 1] + rng0.normal(0.0, 0.15)
    out0 = setar_fit(y0)
    out_b = setar_fit(y)

    c = float(out["c_hat"])
    return {
        "synthetic_c_hat": c,
        "synthetic_c_err": float(abs(c - 0.5)),
        "synthetic_phi_lo": float(out["phi_lo"]),
        "synthetic_phi_hi": float(out["phi_hi"]),
        "synthetic_phi_gap": float(out["phi_lo"] - out["phi_hi"]),
        "synthetic_f_linearity": float(out["f_linearity"]),
        "synthetic_p_linearity": float(p_val),
        "synthetic_null_gap": float(abs(out0["phi_lo"] - out0["phi_hi"])),
        "synthetic_detects": float(
            abs(c - 0.5) < 0.25 and float(out["phi_lo"] - out["phi_hi"]) > 0.25
        ),
        "synthetic_determinism": float(c == float(out_b["c_hat"])),
    }
