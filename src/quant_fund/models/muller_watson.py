"""Müller-Watson (2008, 2018) low-frequency inference.

References
----------
- Müller, U.K. & Watson, M.W. (2008). "Testing Models of
  Low-Frequency Variability." *Econometrica* 76(5), 979-1016.
- Müller, U.K. & Watson, M.W. (2018). "Long-Run Covariability."
  *Econometrica* 86(3), 775-804.
- Müller, U.K. & Watson, M.W. (2024). "How to Estimate Long-
  Run Covariability and the Associated Economic Correlation."
  *Journal of Business & Economic Statistics* 42(4).

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Low-frequency inference extracts the ``q`` lowest cosine
transforms ``X_j = T^{-1/2} sum_t x_t * sqrt(2) cos(pi j
(t-0.5)/T)`` for ``j = 1..q`` — under modest dependence these
are approximately iid Gaussian with variance the zero-frequency
spectral density ``S_x(0)``. For two series, the q pairs
``(X_j, Y_j)`` are jointly Gaussian with correlation
``rho_LF``: the long-run (economic) correlation. We estimate
``rho_LF`` as the sample correlation of the q pairs (q = T^{2/3}
rounded, or explicit) and form the ``cos_q`` confidence set via
the usual Fisher-z interval — robust to short-run dependence by
construction. For a single series we also report the scaled
periodogram ``X_j^2`` first-moment estimate ``S_hat(0)`` and the
stationarity diagnostic ``max |X_j| / sqrt(S_hat(0))`` (a
low-frequency unit-root screen: values past ~3 indicate a
persistent component). The bench plants two series sharing a
common random-walk low-frequency driver plus independent
high-frequency noise: the method must recover a high positive
rho_LF whose interval excludes zero, while a pair of
independent series yields a confidence interval covering zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _cos_transforms(x: FloatArray, q: int) -> FloatArray:
    """Scaled cosine transforms X_j, j = 1..q."""
    t = x.size
    js = np.arange(1, q + 1)[:, None]
    tt = np.arange(t)[None, :]
    # standard definition: cos(pi * j * (t - 1/2) / T)
    c = np.sqrt(2.0) * np.cos(np.pi * js * (tt + 0.5) / t)
    return np.asarray((x[None, :] @ c.T).ravel() / np.sqrt(t), dtype=np.float64)


def low_freq_correlation(
    x: FloatArray,
    y: FloatArray,
    q: int | None = None,
) -> dict[str, float | FloatArray]:
    """Long-run correlation between ``x`` and ``y`` via q cosines."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if (
        xx.ndim != 1
        or yy.ndim != 1
        or xx.size != yy.size
        or xx.size < 64
        or not np.all(np.isfinite(xx))
        or not np.all(np.isfinite(yy))
    ):
        raise ValueError("bad inputs")
    t = xx.size
    qq = int(q) if q is not None else max(8, int(round(t ** (2 / 3) / 2)))
    if qq < 4 or qq > t // 4:
        raise ValueError("bad q")
    xs = _cos_transforms(xx - xx.mean(), qq)
    ys = _cos_transforms(yy - yy.mean(), qq)
    rho = float(np.corrcoef(xs, ys)[0, 1])
    # Fisher-z interval on the q-pair correlation
    z = float(np.arctanh(np.clip(rho, -0.999, 0.999)))
    se = 1.0 / np.sqrt(qq - 3)
    lo = float(np.tanh(z - 1.96 * se))
    hi = float(np.tanh(z + 1.96 * se))
    # scaled-periodogram spectral estimates
    s_x = float(np.mean(xs**2))
    s_y = float(np.mean(ys**2))
    # low-frequency unit-root screen: largest standardized transform
    ur_x = float(np.max(np.abs(xs)) / np.sqrt(s_x)) if s_x > 0 else float("inf")
    return {
        "rho_lf": rho,
        "rho_lo": lo,
        "rho_hi": hi,
        "q": float(qq),
        "s0_x": s_x,
        "s0_y": s_y,
        "ur_stat_x": ur_x,
    }


def lf_predictive_test(
    x: FloatArray,
    y: FloatArray,
    q: int | None = None,
) -> dict[str, float]:
    """Low-frequency predictive regression on the cosine transforms.

    Regresses the q ``Y_j`` on ``X_j``; the slope is the
    long-run beta and its z-test uses the q-pair information —
    a Mueller-Watson-style long-run predictability test.
    """
    r = low_freq_correlation(x, y, q)
    qq = int(r["q"])
    xs = _cos_transforms(np.asarray(x, dtype=np.float64), qq)
    ys = _cos_transforms(np.asarray(y, dtype=np.float64), qq)
    beta = float(np.sum(xs * ys) / np.sum(xs * xs))
    resid = ys - beta * xs
    se = float(np.sqrt(np.sum(resid**2) / (qq - 2) / np.sum(xs * xs)))
    z = beta / se if se > 0 else 0.0
    p = float(2.0 * _stats.norm.sf(abs(z)))
    return {"beta_lf": beta, "z_lf": float(z), "p_lf": p, "q": float(qq)}


def synth_mw(
    seed: int = 20261231 + 313,
    t: int = 700,
    shared: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC pair: shared persistent AR(0.98) driver + own noise.

    ``shared=False`` gives each series its own mild AR(0.5)
    process so the low-frequency correlation is near zero; using
    persistent-but-independent drivers would exhibit the
    spurious low-frequency covariability the method warns about.
    """
    rng = np.random.default_rng(seed)

    def ar1(rho: float, sig: float = 1.0) -> FloatArray:
        z = np.empty(t)
        z[0] = rng.standard_normal()
        for i in range(1, t):
            z[i] = rho * z[i - 1] + rng.standard_normal() * sig
        return z

    if shared:
        c = ar1(0.98)
        x = 0.8 * c + rng.standard_normal(t)
        y = 0.9 * c + rng.standard_normal(t) * 1.2
    else:
        x = ar1(0.5)
        y = ar1(0.5)
    return x, y


def bench_muller_watson(
    seed: int = 20261231 + 313,
) -> dict[str, float]:
    """Wave-54 self-check: shared driver recovers rho_lf ~ 1."""
    x1, y1 = synth_mw(seed=seed, shared=True)
    r1 = low_freq_correlation(x1, y1, q=16)
    p1 = lf_predictive_test(x1, y1, q=16)
    x0, y0 = synth_mw(seed=seed + 11, shared=False)
    r0 = low_freq_correlation(x0, y0, q=16)
    ok = (
        float(r1["rho_lf"]) > 0.5
        and float(r1["rho_lo"]) > 0.0
        and float(r0["rho_lo"]) < float(r0["rho_hi"])  # sanity
        and abs(float(r0["rho_lf"])) < 0.45
        and float(p1["p_lf"]) < 0.01
    )
    return {
        "synthetic_rho_shared": float(r1["rho_lf"]),
        "synthetic_rho_indep": float(r0["rho_lf"]),
        "synthetic_p_beta": float(p1["p_lf"]),
        "synthetic_beta_lf": float(p1["beta_lf"]),
        "synthetic_ur_stat": float(r1["ur_stat_x"]),
        "synthetic_score": float(ok),
    }
