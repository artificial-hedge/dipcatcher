"""Surrogate-data nonlinearity hypothesis tests.

Method-of-surrogates framework (Theiler et al. 1992): generate synthetic
series that preserve the linear structure of the data (spectrum +
distribution via AAFT/IAAFT) but destroy nonlinear dynamics, then
compare a nonlinear discriminating statistic on the data against the
surrogate ensemble. Ships the three classical specification tests for
nonlinearity — BDS (Brock–Dechert–Scheinkman 1986/1996), Keenan (1985),
Tsay (1986).

References
----------
- Theiler, Eubank, Longtin, Galdrikian & Farmer (1992). Testing for
  nonlinearity in time series: the method of surrogate data. *Physica D*
  58:77–94.
- Schreiber & Schmitz (1996). Improved surrogate data for nonlinearity
  tests. *Phys. Rev. Lett.* 77:635–638 (IAAFT).
- Brock, Dechert, Scheinkman & LeBaron (1996). A test for independence
  based on the correlation dimension. *Econometric Reviews* 15:197–235.
- Keenan (1985). A Tukey nonadditivity-type test for time series
  nonlinearity. *Biometrika* 72:39–44.
- Tsay (1986). Nonlinearity tests for time series. *Biometrika*
  73:461–466.

Honesty
-------
Every series here is SYNTHETIC (AR null, GARCH, tent map). Reported
keys are size/power diagnostics of the tests on seeded generators —
they verify the statistics distinguish linear from nonlinear
dynamics, never that any market series is chaotic.

Composition notes
-----------------
- ``metrics/entropy.py``: nonlinear dependence measures (TE, SampEn).
  This module adds the *surrogate null-distribution* machinery and the
  classical econometric nonlinearity tests.
- ``metrics/nonlinearity.py`` / ``spec.py``: conditional-mean tests
  (RESET, White). These are unconditional surrogate-based tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_n: int = 60) -> FloatArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < min_n:
        raise ValueError(f"series too short (n={x.size} < {min_n})")
    if not np.all(np.isfinite(x)):
        raise ValueError("series must be finite")
    return x


def _gaussianize(x: FloatArray) -> FloatArray:
    """Rank-map x onto standard-normal quantiles."""
    ranks = sstats.rankdata(x)
    u = (ranks - 0.5) / x.size
    return np.asarray(sstats.norm.ppf(u))


def aaft_surrogate(x: FloatArray, seed: int = 0) -> FloatArray:
    """AAFT: Gaussianize, phase-randomize, inverse rank-map (Theiler 1992)."""
    x = _as_series(x, min_n=30)
    g = _gaussianize(x)
    spec = np.fft.rfft(g)
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, spec.size))
    phases[0] = 1.0
    if x.size % 2 == 0:
        phases[-1] = 1.0
    gs = np.fft.irfft(spec * phases, n=x.size)
    # inverse rank-map: sorted x values at the ranks of gs
    order = np.argsort(np.argsort(gs))
    return np.asarray(np.sort(x)[order])


def iaaft_surrogate(x: FloatArray, n_iter: int = 50, seed: int = 0) -> FloatArray:
    """Iterative AAFT (Schreiber–Schmitz 1996): alternate between
    spectrum projection and rank-order projection."""
    x = _as_series(x, min_n=30)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    rng = np.random.default_rng(seed)
    s = np.sort(rng.standard_normal(x.size))[np.argsort(np.argsort(rng.permutation(x.size)))]
    s = x[np.argsort(np.argsort(s))]  # start at shuffled-x rank map
    for _ in range(n_iter):
        spec = np.fft.rfft(s)
        s = np.fft.irfft(target_amp * np.exp(1j * np.angle(spec)), n=x.size)
        s = sorted_x[np.argsort(np.argsort(s))]
    return np.asarray(s, dtype=np.float64)


def nonlinear_statistic(x: FloatArray, kind: str = "tser_rev") -> float:
    """Discriminating statistic for surrogate tests.

    ``tser_rev`` — time-reversal asymmetry mean((x_{t+1} − x_t)^3);
    ``skewness`` — |skew|; ``bds`` — BDS stat at m=3, eps=0.7σ.
    """
    x = _as_series(x, min_n=30)
    if kind == "tser_rev":
        return float(np.mean((x[1:] - x[:-1]) ** 3))
    if kind == "skewness":
        return float(abs(sstats.skew(x)))
    if kind == "bds":
        return float(bds_statistic(x, m=3, eps_frac=0.7)[0])
    raise ValueError("kind must be tser_rev|skewness|bds")


def surrogate_test(
    x: FloatArray,
    stat_fn: Callable[[FloatArray], float] | str = "tser_rev",
    n_surr: int = 49,
    method: str = "iaaft",
    seed: int = 0,
) -> float:
    """Two-sided surrogate p-value.

    ``stat_fn`` may be a callable on the series or a
    ``nonlinear_statistic`` kind string. Surrogates are IAAFT (default)
    or AAFT. p = (1 + #|s_surr| ≥ |s_data|) / (n_surr + 1).
    """
    x = _as_series(x)
    if not (n_surr >= 10):
        raise ValueError("n_surr must be >= 10")
    if method not in ("aaft", "iaaft"):
        raise ValueError("method must be 'aaft' or 'iaaft'")
    stat: Callable[[FloatArray], float]
    if isinstance(stat_fn, str):
        stat = partial(nonlinear_statistic, kind=stat_fn)
    else:
        stat = stat_fn
    s_data = stat(x)
    count = 0
    for k in range(n_surr):
        surr = (
            iaaft_surrogate(x, n_iter=30, seed=seed + 10_000 + k)
            if method == "iaaft"
            else aaft_surrogate(x, seed=seed + 10_000 + k)
        )
        if abs(stat(surr)) >= abs(s_data):
            count += 1
    return float((1 + count) / (n_surr + 1))


def _corr_integral(points: FloatArray, eps: float) -> float:
    """Correlation integral C(m, eps) = P(||x_i - x_j||_inf < eps)."""
    n = points.shape[0]
    if n < 2:
        raise ValueError("need >= 2 points")
    d = np.abs(points[:, None, :] - points[None, :, :]).max(axis=2)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(d[iu] < eps))


def _embed(x: FloatArray, m: int, tau: int = 1) -> FloatArray:
    n = x.size - (m - 1) * tau
    if n < 2:
        raise ValueError("embedding leaves too few points")
    idx = np.arange(m) * tau
    return np.asarray(x[np.arange(n)[:, None] + idx[None, :]], dtype=np.float64)


def bds_statistic(
    x: FloatArray, m: int = 3, eps_frac: float = 0.7, max_n: int = 600
) -> tuple[float, float]:
    """BDS statistic and p-value (asymptotic normal).

    Uses the normalized correlation-integral difference
    w = sqrt(n) (C_m(ε) − C_1(ε)^m) / σ_m(ε) with σ_m estimated by the
    iid null standard deviation of the U-statistic (Kanzler-style
    first-order estimate, adequate for SYNTHETIC diagnostics).
    """
    x = _as_series(x)
    if m < 2:
        raise ValueError("m must be >= 2")
    if not eps_frac > 0:
        raise ValueError("eps_frac must be positive")
    x = x[-max_n:]
    sd = float(x.std(ddof=1))
    if sd <= 0:
        raise ValueError("series has zero variance")
    eps = eps_frac * sd
    x = (x - x.mean()) / sd
    c1 = _corr_integral(x[:, None], eps)
    if c1 <= 0:
        raise ValueError("degenerate correlation integral")
    cm = _corr_integral(_embed(x, m), eps)
    diff = cm - c1**m
    # iid-null standard error: σ_m² per BDS (1986) using the 1-d kernel
    k1 = _corr_integral_kernel(x, eps)
    var = _bds_var(m, c1, k1)
    z = math.sqrt(x.size) * diff / math.sqrt(max(var, 1e-16))
    p = float(2.0 * sstats.norm.sf(abs(z)))
    return float(z), p


def _corr_integral_kernel(x: FloatArray, eps: float) -> float:
    """K(ε) = P(|x_i−x_j|<ε AND |x_j−x_l|<ε) — the 1-d overlap kernel."""
    d = np.abs(x[:, None] - x[None, :]) < eps
    n = x.size
    s = d.sum(axis=1) - 1.0  # exclude self
    return float(((s / (n - 1)) ** 2).mean())


def _bds_var(m: int, c1: float, k1: float) -> float:
    """Asymptotic variance of sqrt(n)(C_m − C_1^m) under iid.

    σ_m² = 4 [ K^m + 2 Σ_{j=1}^{m-1} K^{m-j} C^{2j}
              + (m−1)² C^{2m} − m² K C^{2m−2} ],
    with C = C(1, ε) and K = K(ε) both on the unembedded series
    (Brock–Dechert–Scheinkman 1986, eq. for the U-statistic CLT).
    """
    terms = (
        k1**m
        + 2.0 * sum(k1 ** (m - j) * c1 ** (2 * j) for j in range(1, m))
        + (m - 1) ** 2 * c1 ** (2 * m)
        - m**2 * k1 * c1 ** (2 * m - 2)
    )
    return 4.0 * max(terms, 1e-8)


def _ar_residuals(x: FloatArray, m: int) -> tuple[FloatArray, FloatArray]:
    """Fit AR(m) by OLS; return (residuals, fitted values)."""
    x = _as_series(x)
    y = x[m:]
    xreg = np.stack([x[m - i - 1 : x.size - i - 1] for i in range(m)], axis=1)
    xreg = np.column_stack([np.ones(y.size), xreg])
    beta, *_ = np.linalg.lstsq(xreg, y, rcond=None)
    fit = xreg @ beta
    return y - fit, fit


def _f_test(rss0: float, rss1: float, k_extra: int, n2: int) -> tuple[float, float]:
    """F test of k_extra restrictions; n2 = residual df."""
    if n2 <= 0 or rss1 <= 0:
        raise ValueError("degenerate regression")
    f = ((rss0 - rss1) / k_extra) / (rss1 / n2)
    return float(f), float(sstats.f.sf(f, k_extra, n2))


def keenan_test(x: FloatArray, m: int = 4) -> tuple[float, float]:
    """Keenan (1985) one-degree-of-freedom nonadditivity test.

    Stage 1: fit AR(m) to x → fitted ŷ, residuals ê.
    Stage 2: regress ê² on ŷ² → fitted ê̂² = a + b·ŷ².
    Stage 3: augment the AR regression with ê̂² as an extra regressor
    for x_t and F-test its coefficient (Tukey nonadditivity analog).
    """
    x = _as_series(x)
    e, fit = _ar_residuals(x, m)
    y2 = fit**2
    b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(y2.size), y2]), e**2, rcond=None)
    eta = np.column_stack([np.ones(y2.size), y2]) @ b2  # ê̂²
    # stage 3: x_t on [1, m lags, ê̂²]
    yv = x[m:]
    lags = np.column_stack([x[m - i - 1 : x.size - i - 1] for i in range(m)] + [eta])
    xreg = np.column_stack([np.ones(yv.size), lags])
    b, *_ = np.linalg.lstsq(xreg, yv, rcond=None)
    rss1 = float(np.sum((yv - xreg @ b) ** 2))
    xreg0 = np.column_stack([np.ones(yv.size), lags[:, :-1]])
    b0, *_ = np.linalg.lstsq(xreg0, yv, rcond=None)
    rss0 = float(np.sum((yv - xreg0 @ b0) ** 2))
    return _f_test(rss0, rss1, 1, yv.size - xreg.shape[1])


def tsay_test(x: FloatArray, m: int = 4) -> tuple[float, float]:
    """Tsay (1986): quadratic/cross terms of AR fits on residuals."""
    e, _fit = _ar_residuals(x, m)
    y = e[m + 1 :] if e.size > m + 1 else e
    x = _as_series(x)
    # align lagged x to residual support: residuals index t=m..n-1
    lags = np.stack([x[m - i - 1 : x.size - i - 1] for i in range(m)], axis=1)[: y.size]
    quad = [lags[:, i] * lags[:, j] for i in range(m) for j in range(i, m)]
    xreg = np.column_stack([np.ones(y.size), lags, np.stack(quad, axis=1)])
    b, *_ = np.linalg.lstsq(xreg, y, rcond=None)
    rss1 = float(np.sum((y - xreg @ b) ** 2))
    b0, *_ = np.linalg.lstsq(xreg[:, : m + 1], y, rcond=None)
    rss0 = float(np.sum((y - xreg[:, : m + 1] @ b0) ** 2))
    k_extra = len(quad)
    return _f_test(rss0, rss1, k_extra, y.size - xreg.shape[1])


# --- synthetic generators ---------------------------------------------------


def synth_ar(n: int = 600, phi: float = 0.5, seed: int = 0) -> FloatArray:
    """Linear AR(1) — the surrogate null."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def synth_garch(
    n: int = 600, omega: float = 0.05, a: float = 0.1, b: float = 0.85, seed: int = 0
) -> FloatArray:
    """GARCH(1,1) — linear in the mean, nonlinear in the second moment."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    sig2 = np.empty(n)
    sig2[0] = omega / (1 - a - b)
    z = rng.standard_normal(n)
    for i in range(n):
        if i > 0:
            sig2[i] = omega + a * x[i - 1] ** 2 + b * sig2[i - 1]
        x[i] = math.sqrt(sig2[i]) * z[i]
    return x


def synth_tent(n: int = 600, mu: float = 1.99, seed: int = 0) -> FloatArray:
    """Tent map — deterministic chaos."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.uniform(0.01, 0.99)
    for i in range(1, n):
        x[i] = mu * min(x[i - 1], 1 - x[i - 1])
    return x


def bench_surrogate_nonlinear(seed: int = 20261231 + 155) -> dict[str, float]:
    """SYNTHETIC surrogate-test telemetry (size + power)."""
    # BDS is an iid-ness test: size is measured on AR residuals (linear
    # dependence removed), power on the raw tent map
    n_bds_rep = 6
    ar_rej, tent_rej = 0, 0
    for r in range(n_bds_rep):
        res_ar, _fit = _ar_residuals(synth_ar(400, seed=seed + r), m=2)
        _, p_ar = bds_statistic(res_ar, m=3)
        _, p_t = bds_statistic(synth_tent(400, seed=seed + 100 + r), m=3)
        ar_rej += int(p_ar < 0.05)
        tent_rej += int(p_t < 0.05)
    # Keenan/Tsay power: on GARCH (pure second-moment — these are
    # conditional-MEAN tests, so honest power is low) and on bilinear
    # (their design target — should reject)
    k_rej = t_rej = kb_rej = tb_rej = 0
    n_kt = 6
    for r in range(n_kt):
        g = synth_garch(500, seed=seed + 200 + r)
        _f, pk = keenan_test(g, m=4)
        _f2, pt = tsay_test(g, m=4)
        k_rej += int(pk < 0.05)
        t_rej += int(pt < 0.05)
        rng_b = np.random.default_rng(seed + 400 + r)
        eb = rng_b.standard_normal(500)
        xb = np.zeros(500)
        for tt in range(2, 500):
            xb[tt] = 0.9 * eb[tt - 1] * xb[tt - 2] + eb[tt]
        _f3, pkb = keenan_test(xb, m=4)
        _f4, ptb = tsay_test(xb, m=4)
        kb_rej += int(pkb < 0.05)
        tb_rej += int(ptb < 0.05)
    # surrogate test: tent detected, AR not
    p_tent = surrogate_test(synth_tent(400, seed=seed + 300), "tser_rev", n_surr=29, seed=seed + 1)
    p_ar_surr = surrogate_test(synth_ar(400, seed=seed + 301), "tser_rev", n_surr=29, seed=seed + 2)
    # AAFT spectral preservation
    xs = synth_ar(400, phi=0.8, seed=seed + 9)
    surr = aaft_surrogate(xs, seed=0)
    psd_x = np.abs(np.fft.rfft(xs)) ** 2
    psd_s = np.abs(np.fft.rfft(surr)) ** 2
    corr = float(np.corrcoef(psd_x[1:], psd_s[1:])[0, 1])
    # determinism
    det = float(np.array_equal(aaft_surrogate(xs, seed=5), aaft_surrogate(xs, seed=5)))
    return {
        "synthetic_bds_size_ar": ar_rej / n_bds_rep,
        "synthetic_bds_power_tent": tent_rej / n_bds_rep,
        "synthetic_keenan_power_garch": k_rej / n_kt,
        "synthetic_tsay_power_garch": t_rej / n_kt,
        "synthetic_keenan_power_bilinear": kb_rej / n_kt,
        "synthetic_tsay_power_bilinear": tb_rej / n_kt,
        "synthetic_surr_p_tent": p_tent,
        "synthetic_surr_p_ar": p_ar_surr,
        "synthetic_aaft_psd_corr": corr,
        "synthetic_determinism": det,
    }
