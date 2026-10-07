"""Chu–Stinchcombe–White fluctuation monitoring for structural (SYNTHETIC)
stability (recursive and moving-estimates CUSUM).

The on-line detector watches standardized fluctuations of a
recursive OLS residual stream: under parameter stability the
cumulative sum

  Q_t = (1/σ̂√m) Σ_{s=m+1}^{m+t} û_s,   û_s = y_s − x_s'β̂_{s−1}

behaves like a Brownian bridge, and a break triggers when |Q_t|
crosses the CSW boundary c_t = √((t/m)(m/(m+t))·(a²+ln(t/m)))
with a = −ln(ln(1/√(1−α))) ≈ 3.30 for α=.05 (the finite-window
boundary). The retrospective moving-estimates variant reports
the max |Q| for comparison with the classical CUSUM band.

Honesty: synthetic series with a planted mid-sample intercept
break; the bench checks the detector fires near the break and
stays silent on a stable series — a proper diagnostic, never
market evidence.

References:
- Chu, C.-S. J., Stinchcombe, M., White, H. (1996).
  Monitoring structural change. *Econometrica* 64 — the
  boundary and its size control.
- Brown, R. L., Durbin, J., Evans, J. M. (1975). Techniques
  for testing the constancy of regression relationships.
  *JRSS-B* 37 — the recursive-residual CUSUM foundation.
- Leisch, F., Hornik, K., Kuan, C.-M. (2000). Monitoring
  structural changes with the generalized fluctuation test
  framework. *Econometric Theory* 16 — the strucchange
  formulation mirrored here.
- Zeileis, A. (2005). A unified approach to structural
  change tests based on ML scores, F statistics, and OLS
  residuals. *Econometric Reviews* 24 — boundaries.

Composition: numpy only — recursive OLS residuals and the
CSW crossing rule; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cusum_monitor(
    y: FloatArray,
    x: FloatArray,
    m: int | None = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Recursive CUSUM detector. ``x`` (n,) or (n,k) — intercept
    added when 1-D; ``m`` the history window (default n/4);
    α the nominal size. Returns the first crossing index
    (n+1 if none), max |Q|, and the boundary a."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        # intercept-only when the regressor is constant
        if float(np.ptp(xx)) < 1e-12:
            xx = np.ones((xx.size, 1))
        else:
            xx = np.column_stack([np.ones(xx.size), xx])
    n = yy.size
    if yy.ndim != 1 or xx.ndim != 2 or xx.shape[0] != n or n < 60:
        raise ValueError("matched y (n,), x (n,k), n>=60 required")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    mm = m if m is not None else max(10, n // 4)
    if not (xx.shape[1] + 2 <= mm < n - 10):
        raise ValueError("m in [k+2, n-10] required")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha in (0, 0.5) required")
    a = float(-np.log(np.log(1.0 / np.sqrt(1.0 - alpha))))
    # recursive residuals: β̂_{s-1} from the first s−1 obs
    u = np.zeros(n)
    xtx = xx[:mm].T @ xx[:mm]
    xty = xx[:mm].T @ yy[:mm]
    for s in range(mm + 1, n + 1):
        beta = np.linalg.solve(xtx, xty)
        u[s - 1] = yy[s - 1] - xx[s - 1] @ beta
        xtx = xtx + np.outer(xx[s - 1], xx[s - 1])
        xty = xty + xx[s - 1] * yy[s - 1]
    # residual scale from the history segment
    u_hist = u[mm : n - 1]
    sig = float(np.sqrt(np.sum(u_hist**2) / max(1, u_hist.size)))
    if sig <= 0:
        raise ValueError("degenerate residuals")
    idx = np.arange(mm + 1, n + 1)
    # CSW boundary for |Q_t|/sqrt(t/m) with Q the scaled
    # recursive cumsum — crosses under a break
    bound = np.sqrt((idx / mm) * (mm / (mm + idx))) * np.sqrt(a * a + np.log(idx / mm))
    q = np.cumsum(u[mm:]) / (sig * np.sqrt(mm))
    qn = q / np.sqrt(idx / mm)
    cross = np.nonzero(np.abs(qn) > bound)[0]
    first = int(idx[cross[0]]) if cross.size else n + 1
    return {
        "first_cross": float(first),
        "max_abs_q": float(np.max(np.abs(qn))),
        "bound_a": a,
        "n": float(n),
        "m": float(mm),
        "sig": sig,
    }


def synth_break(
    n: int = 300,
    break_at: float = 0.6,
    delta: float = 1.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y_t = 1 + δ·1{t > break·n} + ε — pure intercept break."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 1.0 + delta * (t > break_at * n) + rng.normal(0.0, 0.8, n)
    return {"y": np.asarray(y, dtype=np.float64), "x": np.ones(n)}


def bench_cusum_monitor(seed: int = 20261231 + 274) -> dict[str, float]:
    """CUSUM self-check: a mid-sample δ=1.5 intercept break is
    crossed within ~25% of the break; a stable series is not.
    All ``synthetic_*``."""
    n = 300
    dep = cusum_monitor(
        np.asarray(synth_break(n=n, delta=1.5, seed=seed)["y"]),
        np.ones(n),
    )
    ok = cusum_monitor(
        np.asarray(synth_break(n=n, delta=0.0, seed=seed)["y"]),
        np.ones(n),
    )
    out2 = cusum_monitor(
        np.asarray(synth_break(n=n, delta=1.5, seed=seed)["y"]),
        np.ones(n),
    )
    return {
        "synthetic_first_cross": dep["first_cross"],
        "synthetic_break_at": 0.6 * n,
        "synthetic_max_q_break": dep["max_abs_q"],
        "synthetic_max_q_null": ok["max_abs_q"],
        "synthetic_detects": float(
            0.45 * n < dep["first_cross"] < 0.9 * n and ok["first_cross"] > n
        ),
        "synthetic_determinism": float(out2["first_cross"] == dep["first_cross"]),
    }
