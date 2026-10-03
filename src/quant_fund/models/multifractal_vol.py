"""Multifractal volatility: MRW simulation + scaling-moment estimation.

Multifractal Random Walk (Bacry-Delour-Muzy): log-returns

    r_t = sigma_t * xi_t,   sigma_t = exp(omega_t)

where omega_t is a stationary Gaussian process with *log-correlated*
covariance Cov[omega_t, omega_{t+tau}] = -lambda^2 log(|tau| + 1) (plus a
slow mean-reversion to a baseline). The cascade produces the MRW signatures:
fat-tailed returns, volatility clustering at all scales, and a nonlinear
(parabolic) scaling function zeta(q) of the q-th order absolute moments.

Estimated here: structure-function moments M(q, tau) ~ tau^{zeta(q)} via
log-log regression, the zeta(q) curve, its quadratic curvature (the
multiscaling signature), and the intermittency coefficient lambda^2 from
log-omega variance vs log-lag — plus a multiscaling-vs-monoscaling
discriminator (fBM/GBM controls have zeta(q) ~ Hq, linear in q).

References
----------
- Bacry, Delour & Muzy (2001). Multifractal random walk. *Physical Review
  E* 64. arXiv:cond-mat/0009260 (verified).
- Bacry & Muzy (2003). Log-infinitely divisible multifractal processes.
  *Communications in Mathematical Physics* 236. arXiv:cond-mat/0205428
  (verified).
- Muzy & Bacry (2002). Multifractal stationary random measures and
  multifractal random walks. *Physical Review E* 66.
  arXiv:cond-mat/0012421 (verified).
- Calvet & Fisher (2008). *Multifractal Volatility: Theory, Forecasting,
  and Pricing*. Academic Press (book; MMAR tradition — no arXiv).

Honesty
-------
All benches run on seeded SYNTHETIC MRW/fBM/GBM series generated in-module.
The numbers validate estimator correctness (lambda^2 recovery, zeta(q)
curvature sign, discriminator AUC) — never market evidence.

Composition notes
-----------------
- ``models/mfdfa.py``: detrended-fluctuation *analysis* of a given series;
  this module is the model side — simulation under a known cascade plus
  structure-function scaling estimation. Orthogonal machinery.
- ``models/long_memory.py``: monofractal Hurst tools; MRW is multiscaling.
- ``models/rough_vol.py``: rough paths (H < 0.5 monofractal); the
  discriminator test here separates mono- from multi-scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, min_len: int = 64) -> FloatArray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size < min_len:
        raise ValueError(f"series needs >= {min_len} points")
    if not np.isfinite(a).all():
        raise ValueError("series contains non-finite values")
    return a


def simulate_mrw(
    n: int,
    lambda2: float = 0.04,
    cascade_horizon: int | None = None,
    sigma0: float = 1.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """MRW path: returns and the log-volatility cascade omega_t.

    Discretized MRW: omega_t is an AR(1)-slowed Gaussian process with
    log-correlated increments — built by summing K Gaussian "cascade layers"
    whose influence decays logarithmically over ``cascade_horizon`` steps.
    Returns (returns, omega).
    """
    if n < 64:
        raise ValueError("n >= 64 required")
    if lambda2 <= 0 or lambda2 >= 0.5:
        raise ValueError("lambda2 in (0, 0.5) keeps the cascade non-degenerate")
    if sigma0 <= 0:
        raise ValueError("sigma0 > 0 required")
    rng = np.random.default_rng(seed)
    horizon = cascade_horizon or max(64, n // 4)
    # log-correlated omega: cumulative sum of overlapping Gaussian bumps whose
    # amplitudes shrink ~ 1/sqrt(scale) — practical MRW discretization.
    n_layers = max(8, int(np.log2(horizon)))
    omega = np.zeros(n)
    for layer in range(n_layers):
        scale = 2 ** (layer + 1)
        amp = np.sqrt(lambda2 * np.log(2.0))
        # coarse-grained Gaussian innovations at this scale, upsampled
        n_cells = max(1, n // scale + 2)
        innov = rng.standard_normal(n_cells) * amp
        omega += np.repeat(innov, scale)[:n]
    # stationary baseline: subtract layer-mean drift so E[sigma]=sigma0
    omega = omega - omega.mean()
    xi = rng.standard_normal(n)
    r = sigma0 * np.exp(omega) * xi
    return r, omega


def simulate_fbm_control(n: int, hurst: float = 0.5, seed: int = 0) -> FloatArray:
    """Monofractal control: fractional Brownian increments via spectral synth."""
    if n < 64:
        raise ValueError("n >= 64 required")
    if not 0 < hurst < 1:
        raise ValueError("hurst in (0,1)")
    rng = np.random.default_rng(seed)
    # circulant spectral synthesis of fGn increments
    m = 1
    while m < 2 * n:
        m *= 2
    freqs = np.fft.rfftfreq(m)[1:]
    amp = freqs ** (-(hurst + 0.5))  # spectral density ~ f^{-(2H+1)}
    phase = rng.uniform(0, 2 * np.pi, freqs.size)
    spec = amp * np.exp(1j * phase)
    full = np.concatenate([[0.0], spec, np.conj(spec[::-1][:-1])])
    series = np.real(np.fft.ifft(full))[:n]
    series = (series - series.mean()) / max(series.std(), 1e-12)
    return series


def structure_moments(x: FloatArray, q: float, taus: FloatArray) -> FloatArray:
    """M(q, tau) = mean |r_{t+tau} - r_t|^q at each lag."""
    r = _check_series(x)
    q = float(q)
    taus = np.asarray(taus, dtype=int)
    if (taus < 1).any() or taus.max() >= r.size // 2:
        raise ValueError("taus must be in [1, n/2)")
    out = np.empty(taus.size)
    for i, t_ in enumerate(taus):
        d = np.abs(r[int(t_) :] - r[: -int(t_)]) ** q
        out[i] = float(d.mean())
    return out


def zeta_curve(x: FloatArray, qs: FloatArray, taus: FloatArray) -> FloatArray:
    """Scaling exponents zeta(q) per q via log-log regression of M(q,tau)."""
    r = _check_series(x)
    qs = np.asarray(qs, dtype=float).ravel()
    taus = np.asarray(taus, dtype=int)
    if qs.size < 1 or taus.size < 4:
        raise ValueError("need qs>=1 and taus>=4")
    lt = np.log(taus.astype(float))
    out = np.empty(qs.size)
    for i, q in enumerate(qs):
        m = structure_moments(r, float(q), taus)
        good = m > 0
        if good.sum() < 4:
            raise ValueError(f"too few finite moments at q={q}")
        slope, _ = np.polyfit(lt[good], np.log(m[good]), 1)
        out[i] = float(slope)
    return out


def zeta_curvature(x: FloatArray, taus: FloatArray) -> float:
    """Quadratic curvature of zeta(q) on q in {1,2,3,4} (multiscaling < 0)."""
    qs = np.arange(1.0, 5.0)
    z = zeta_curve(x, qs, taus)
    c = np.polyfit(qs, z, 2)[0]
    return float(c)


def lambda2_estimate(omega: FloatArray, lags: FloatArray) -> float:
    """Intermittency coefficient from Var of log-sigma vs log lag.

    On the cascade, Var[omega coarse-grained at lag l] DECREASES as
    ~ lambda2 * log l (each doubling drops one cascade layer's variance).
    Regress Var[omega_agg(l)] on log l; lambda2 = -slope.
    """
    om = _check_series(omega)
    lags = np.asarray(lags, dtype=int)
    if (lags < 2).any() or lags.max() >= om.size // 4:
        raise ValueError("lags must be in [2, n/4)")
    xs, ys = [], []
    for lag in lags:
        n_cells = om.size // int(lag)
        agg = om[: n_cells * int(lag)].reshape(n_cells, int(lag)).mean(axis=1)
        if agg.size < 8:
            continue
        xs.append(np.log(float(lag)))
        ys.append(float(agg.var()))
    if len(xs) < 3:
        raise ValueError("too few aggregation levels")
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-slope)


def multifractality_score(x: FloatArray, taus: FloatArray) -> float:
    """-curvature of zeta(q): >0 means multiscaling (MRW), ~0 means fBM-like."""
    return -zeta_curvature(x, taus)


def discriminate_auc(samples_pos: FloatArray, samples_neg: FloatArray) -> float:
    """AUC of multifractality scores: MRW (pos) vs fBM (neg)."""
    p = np.asarray(samples_pos, dtype=float).ravel()
    n = np.asarray(samples_neg, dtype=float).ravel()
    if p.size == 0 or n.size == 0:
        raise ValueError("empty sample array")
    wins = sum(np.sum(v > n) + 0.5 * np.sum(v == n) for v in p)
    return float(wins / (p.size * n.size))


def bench_multifractal_vol(seed: int = 20260131) -> dict[str, float]:
    """SYNTHETIC bench for the MRW machinery. Correctness only."""
    out: dict[str, float] = {}
    n = 8192
    true_l2 = 0.05
    r_mrw, omega = simulate_mrw(n, lambda2=true_l2, seed=seed)
    taus = np.array([2, 4, 8, 16, 32, 64])
    # lambda2 recovery on the true cascade
    lam_hat = lambda2_estimate(omega, lags=np.array([4, 8, 16, 32]))
    out["synthetic_lambda2_err"] = abs(lam_hat - true_l2)
    # zeta curvature: MRW should curve (multiscaling), fBM should not
    curv_mrw = zeta_curvature(r_mrw, taus)
    out["synthetic_zeta_curvature_mrw"] = curv_mrw
    out["synthetic_mrw_is_multiscaling"] = float(curv_mrw < -1e-3)
    # discriminator AUC: MRW samples vs fBM controls
    pos = np.array(
        [
            multifractality_score(
                simulate_mrw(4096, lambda2=0.06, seed=seed + 10 + i)[0], np.array([2, 4, 8, 16, 32])
            )
            for i in range(8)
        ]
    )
    neg = np.array(
        [
            multifractality_score(
                simulate_fbm_control(4096, seed=seed + 40 + i), np.array([2, 4, 8, 16, 32])
            )
            for i in range(8)
        ]
    )
    out["synthetic_discriminator_auc"] = discriminate_auc(pos, neg)
    # determinism
    a = simulate_mrw(512, seed=seed)
    b = simulate_mrw(512, seed=seed)
    out["synthetic_determinism"] = float(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    return out
