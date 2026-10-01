"""Marčenko–Pastur covariance denoising via random matrix theory.

Wave-26 lane: random-matrix-theory tools for separating signal from noise in
a sample covariance matrix ``S = X.T @ X / n_obs`` (``X`` ``(n_obs, p)``
centered), where the limiting spectral distribution under the null
``Sigma = sigma^2 * I`` is the Marčenko–Pastur law with aspect ratio
``q = p / n_obs``.

What is implemented here, keyed to the literature:

1. ``mp_density`` / ``mp_bounds`` — the MP density
   ``f(x) = sqrt((lam+ - x)(x - lam-)) / (2 pi q sigma^2 x)`` supported on
   ``[lam-, lam+]`` with ``lam± = sigma^2 (1 ± sqrt(q))^2`` (Marčenko & Pastur
   1967). The density integrates to ``min(1, 1/q)``: for ``q > 1`` the law
   carries an atom ``(1 - 1/q)`` at zero, so the continuous part is a
   defective density; ``mp_density`` returns the continuous part only.
2. ``mp_edge_test`` — share of sample eigenvalues above the upper MP edge
   ``lam+`` (signal detection à la Laloux/Plerou) plus a Tracy–Widom "soft"
   edge: under the null the largest eigenvalue fluctuates around ``lam+`` as
   ``lam_max ≈ mu_J + sigma_J * TW_1`` with Johnstone's (2001) normalization
   ``mu_J = sigma^2 (sqrt(n-1) + sqrt(p))^2 / n`` and
   ``sigma_J = sigma^2 (sqrt(n-1) + sqrt(p)) (1/sqrt(n-1) + 1/sqrt(p))^{1/3}
   / n``. The soft edge is ``mu_J + z * sigma_J`` at a chosen TW_1 quantile
   ``z`` (default ≈ 95th percentile, so a pure-noise spectrum breaches it in
   ~5% of draws); eigenvalues between the hard edge and the soft edge are
   noise-level fluctuations, not tradable signal.
3. ``eigenvalue_clip`` — the Laloux/Bouchaud eigenvalue-clipping RIE: every
   eigenvalue inside the MP bulk (``λ_i ≤ lam+``) is replaced by the bulk
   mean ``(tr(S) - Σ_{λ_i > lam+} λ_i) / n_bulk``, preserving the trace;
   eigenvectors and the above-edge signal block are kept. The noise
   variance ``sigma^2`` is estimated from the spectrum by a bounded
   fixed-point on ``sigma^2 = mean(λ_i ≤ sigma^2 (1+sqrt(q))^2)`` seeded at
   the median eigenvalue (robust when most eigenvalues are bulk).
4. ``rotation_shrink`` — rotation-equivariant linear shrinkage of the
   eigenvalues toward the scaled-identity target ``mu * I``,
   ``mu = tr(S)/p``: ``λ_i' = (1 - alpha) λ_i + alpha * mu``, eigenvectors
   unchanged. With ``alpha=None`` the intensity is the oracle-approximating
   shrinkage (OAS) closed form of Chen, Wiesel, Eldar & Hero (2010, eq. 23,
   large-p limit): ``alpha_hat = min(1, (p^2 tr(S^2) + p tr(S)^2) /
   ((n + 1) (p^2 tr(S^2) - tr(S)^2)))``, a function of the spectrum alone —
   no raw data needed, unlike the Ledoit–Wolf/MMSE estimators in
   ``models.covariance`` which need ``X`` for the fourth-moment terms.
5. ``sim_correlated`` — seeded SYNTHETIC factor-structured Gaussian data
   ``X = F (B s)^T + noise * Z`` with ``F`` ``(n_obs, k)`` iid N(0,1),
   ``B`` ``(p, k)`` iid N(0,1) row-normalized to ``||B_j|| = sqrt(k)``,
   ``s = sqrt((1 - noise^2) / k)`` and ``Z`` iid N(0,1), so each feature has
   exactly unit population variance and the population covariance is
   ``Sigma = s^2 B B^T + noise^2 I`` — ``k`` planted spikes on a ``noise^2``
   MP bulk.
6. ``bench_marchenko_pastur`` — float-only SYNTHETIC correctness bench:
   signal-count error, edge violations on pure noise, Frobenius-norm
   improvement of the two denoised estimators over the raw sample
   covariance, trace preservation, OAS intensity, and determinism. Every
   key is ``synthetic_*``.

Composition notes (nothing here modifies or duplicates them): this module
works from the sample-covariance *spectrum* alone, complementing
``models.covariance`` (sklearn ``LedoitWolf``/``OAS`` wrappers that require
the raw returns matrix, plus ``repair_psd``), ``models.robust_cov``
(FastMCD/OGK high-breakdown estimators on raw data), and
``models.factor_models`` (``pca_factors`` / ``bai_ng_factors`` /
``factor_residuals`` — cross-sectional factor structure the denoised
covariance feeds). ``pipeline.forecast.covariance`` owns the optimizer-facing
covariance surface; this lane is a standalone metrics diagnostic.

Honesty: every number here is a SYNTHETIC correctness diagnostic computed on
seeded Gaussian factor worlds — never market evidence, never a headline
metric. ``bench_marchenko_pastur`` emits only ``synthetic_*`` float keys
(the AGENTS.md honesty contract); no Sharpe/Sortino/Calmar/P&L/NAV tokens
appear anywhere in its output. Research-diagnostic only;
``live_pnl_claim`` is False everywhere it could apply.

References (arXiv identifiers verified against arxiv.org/abs/* on
2026-10-01 — the lane brief's ids ``1602.03570``, ``math/9811155`` resolve
to unrelated papers and ``0907.4698`` is Chen–Wiesel–Eldar–Hero (OAS), not
Ledoit–Wolf; corrected here):
- Marčenko, V.A. & Pastur, L.A. (1967). "Distribution of eigenvalues for
  some sets of random matrices." Mathematics of the USSR-Sbornik
  1(4):457-483 — the MP law and its support.
- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
  large-dimensional covariance matrices." Journal of Multivariate Analysis
  88(2):365-411 — linear shrinkage toward scaled identity.
- Ledoit, O. & Wolf, M. (2012). "Nonlinear shrinkage estimation of
  large-dimensional covariance matrices." Annals of Statistics
  40(2):1024-1060, arXiv:1207.5322 — the nonlinear RIE this lane's clipping
  approximates.
- Chen, Y., Wiesel, A., Eldar, Y.C. & Hero, A.O. (2010). "Shrinkage
  algorithms for MMSE covariance estimation." IEEE Transactions on Signal
  Processing 58(10):5016-5029, arXiv:0907.4698 — the OAS intensity used by
  ``rotation_shrink``.
- Tracy, C.A. & Widom, H. (1994). "Level-spacing distributions and the
  Airy kernel." Communications in Mathematical Physics 159:151-174; and
  Tracy & Widom (1997), "The distribution of the largest eigenvalue in the
  Gaussian ensembles", arXiv:solv-int/9707001 — the edge-fluctuation law.
- Johnstone, I.M. (2001). "On the distribution of the largest eigenvalue
  in principal components analysis." Annals of Statistics 29(2):295-327 —
  the Wishart centering/scaling ``mu_J``, ``sigma_J`` used in the soft edge.
- Bun, J., Bouchaud, J.-P. & Potters, M. (2017). "Cleaning large
  correlation matrices: tools from random matrix theory." Physics Reports
  666:1-109, arXiv:1610.08104 — review of eigenvalue clipping and
  rotationally invariant estimators.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Tracy–Widom F_1 reference constant: the ~95th-percentile quantile used for
# the soft edge — a pure-noise largest eigenvalue crosses it in ~5% of draws.
TW1_Q95 = 0.9793

_SIGMA2_FIXPOINT_TOL = 1e-12
_SIGMA2_FIXPOINT_MAX_ITER = 100
_BENCH_N_OBS = 250
_BENCH_N_FEAT = 50
_BENCH_K_FACTORS = 5
_BENCH_NOISE = 0.6
_BENCH_GRID = 4001


def _check_q(q: float) -> float:
    qq = float(q)
    if not np.isfinite(qq) or qq <= 0.0:
        raise ValueError("q must be a finite positive aspect ratio")
    return qq


def _check_sigma2(sigma2: float) -> float:
    s2 = float(sigma2)
    if not np.isfinite(s2) or s2 <= 0.0:
        raise ValueError("sigma2 must be a finite positive variance")
    return s2


def _check_n_obs(n_obs: int) -> int:
    if isinstance(n_obs, bool) or not isinstance(n_obs, (int, np.integer)):
        raise ValueError("n_obs must be an integer")
    n = int(n_obs)
    if n < 2:
        raise ValueError("n_obs must be >= 2")
    return n


def _check_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _check_cov(cov: Array) -> Array:
    s = np.asarray(cov, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("cov must contain only finite values")
    return np.asarray(0.5 * (s + s.T), dtype=np.float64)


def _eigh(cov: Array) -> tuple[Array, Array]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    return (
        np.asarray(eigvals, dtype=np.float64),
        np.asarray(eigvecs, dtype=np.float64),
    )


def _rebuild(eigvecs: Array, eigvals: Array) -> Array:
    out = (eigvecs * eigvals) @ eigvecs.T
    return np.asarray(0.5 * (out + out.T), dtype=np.float64)


def _estimate_sigma2(eigvals: Array, q: float) -> float:
    """Fixed-point bulk-variance estimate from the spectrum.

    Iterates ``sigma2 <- mean(eigvals <= sigma2 (1 + sqrt(q))^2)`` starting
    from the median eigenvalue: under the MP null the median is interior to
    the bulk, and planted spikes sit far above the edge, so the iteration
    climbs until the bulk is covered and then holds at the self-consistent
    bulk mean. Deterministic and bounded; if the included set ever empties
    the iterate freezes (pathological spectra where every eigenvalue sits
    above the edge of the running estimate).
    """
    if eigvals.size == 0:
        raise ValueError("eigvals must be non-empty")
    sigma2 = float(np.median(eigvals))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("cannot estimate sigma2 from a non-positive median eigenvalue")
    for _ in range(_SIGMA2_FIXPOINT_MAX_ITER):
        edge = sigma2 * (1.0 + math.sqrt(q)) ** 2
        bulk = eigvals[eigvals <= edge]
        if bulk.size == 0:
            break
        nxt = float(np.mean(bulk))
        if abs(nxt - sigma2) <= _SIGMA2_FIXPOINT_TOL * sigma2:
            sigma2 = nxt
            break
        sigma2 = nxt
    return sigma2


def _oas_alpha(cov: Array, n_obs: int) -> float:
    """OAS shrinkage intensity of ``cov`` (spectrum-only, eq. 23 large-p)."""
    s = np.asarray(cov, dtype=np.float64)
    p = s.shape[0]
    tr_s = float(np.trace(s))
    tr_s2 = float(np.sum(s * s))
    num = p * p * tr_s2 + p * tr_s * tr_s
    den = (n_obs + 1.0) * (p * p * tr_s2 - tr_s * tr_s)
    return 1.0 if den <= 0.0 else min(max(num / den, 0.0), 1.0)


def mp_density(x: Array, q: float, sigma2: float = 1.0) -> Array:
    """Marchenko–Pastur density of the sample spectrum at ``x``.

    ``f(x) = sqrt((lam+ - x)(x - lam-)) / (2 pi q sigma^2 x)`` on the support
    ``[lam-, lam+]``, zero outside (elementwise, vectorized). At ``q = 1``
    the lower edge is 0 and the density diverges like ``1/sqrt(x)``; the
    value at exactly ``x = 0`` is returned as ``+inf`` consistently with the
    formula's limit. For ``q > 1`` this is the continuous part of the law
    only (total mass ``1/q < 1``; the remainder is an atom at zero).
    """
    qq = _check_q(q)
    s2 = _check_sigma2(sigma2)
    xa = np.asarray(x, dtype=np.float64)
    if not np.isfinite(xa).all():
        raise ValueError("x must contain only finite values")
    lam_min, lam_max = mp_bounds(qq, s2)
    inside = (xa >= lam_min) & (xa <= lam_max)
    out = np.zeros(xa.shape, dtype=np.float64)
    num = np.sqrt(np.maximum((lam_max - xa) * (xa - lam_min), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = num / (2.0 * math.pi * qq * s2 * xa)
    out[inside] = dens[inside]
    # q = 1 ⇒ lam_min = 0 ⇒ the density's 1/sqrt(x) limit at x = 0 is +inf.
    out[inside & (xa == 0.0)] = np.inf
    return np.asarray(out, dtype=np.float64)


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP support edges ``sigma^2 (1 -/+ sqrt(q))^2`` as ``(lam_min, lam_max)``."""
    qq = _check_q(q)
    s2 = _check_sigma2(sigma2)
    sq = math.sqrt(qq)
    return s2 * (1.0 - sq) ** 2, s2 * (1.0 + sq) ** 2


def mp_edge_test(
    eigvals: Array,
    q: float,
    n_obs: int,
    sigma2: float | None = None,
    tw_quantile: float = TW1_Q95,
) -> dict[str, float]:
    """Share of eigenvalues above the MP edge, with a Tracy–Widom soft edge.

    ``q = p / n_obs`` is the aspect ratio of the sample matrix the
    eigenvalues came from; ``n_obs`` sets the Johnstone finite-sample
    centering/scaling of the largest eigenvalue (with ``p`` taken as the
    spectrum length). ``sigma2`` is the bulk noise variance; when ``None``
    it is estimated from the spectrum via the bounded median-seeded fixed
    point also used by ``eigenvalue_clip``.

    Returns a float-only dict: ``edge`` / ``edge_tw`` (hard and soft upper
    edges), ``share_above_edge`` / ``share_above_tw_edge`` (the detection
    shares — eigenvalues between them are noise-level fluctuations, not
    signal), ``count_above_edge``, ``tw_scale`` (the ``sigma_J`` width of
    the soft-edge band), ``sigma2`` (the bulk variance used) and
    ``bulk_mean`` (mean of eigenvalues at or below the hard edge).
    """
    qq = _check_q(q)
    n = _check_n_obs(n_obs)
    z = float(tw_quantile)
    if not np.isfinite(z):
        raise ValueError("tw_quantile must be finite")
    ev = np.asarray(eigvals, dtype=np.float64)
    if ev.ndim != 1 or ev.size == 0:
        raise ValueError("eigvals must be a non-empty 1-D array")
    if not np.isfinite(ev).all():
        raise ValueError("eigvals must contain only finite values")
    s2 = _estimate_sigma2(ev, qq) if sigma2 is None else _check_sigma2(sigma2)

    p = ev.size
    _, lam_max = mp_bounds(qq, s2)

    # Johnstone (2001) Wishart edge normalization for the (1/n) X'X
    # convention: lam_max_hat ≈ mu_J + sigma_J * TW_1.
    sqrt_n = math.sqrt(n - 1.0)
    sqrt_p = math.sqrt(float(p))
    mu_j = s2 * (sqrt_n + sqrt_p) ** 2 / n
    sigma_j = s2 * (sqrt_n + sqrt_p) / n * (1.0 / sqrt_n + 1.0 / sqrt_p) ** (1.0 / 3.0)
    edge_tw = mu_j + z * sigma_j

    above = ev > lam_max
    above_tw = ev > edge_tw
    bulk = ev[~above]
    return {
        "edge": lam_max,
        "edge_tw": edge_tw,
        "tw_scale": sigma_j,
        "sigma2": s2,
        "count_above_edge": float(np.count_nonzero(above)),
        "share_above_edge": float(np.mean(above.astype(np.float64))),
        "share_above_tw_edge": float(np.mean(above_tw.astype(np.float64))),
        "bulk_mean": float(np.mean(bulk)) if bulk.size else float("nan"),
    }


def eigenvalue_clip(cov: Array, n_obs: int) -> Array:
    """Denoise a sample covariance by MP-bulk eigenvalue clipping.

    Every eigenvalue at or below the upper MP edge ``sigma^2 (1+sqrt(q))^2``
    is replaced by the bulk mean ``(tr(S) - sum signal block) / n_bulk`` so
    the trace is preserved exactly; eigenvectors and the signal block are
    untouched. ``sigma^2`` is estimated from the spectrum (median-seeded
    fixed point). Returns the rebuilt symmetric PSD covariance as a
    ``np.float64`` array.
    """
    s = _check_cov(cov)
    n = _check_n_obs(n_obs)
    eigvals, eigvecs = _eigh(s)
    p = s.shape[0]
    q = p / float(n)
    sigma2 = _estimate_sigma2(eigvals, q)
    _, lam_max = mp_bounds(q, sigma2)
    bulk = eigvals <= lam_max
    if not np.any(bulk):
        # Degenerate spectrum: the running edge is below every eigenvalue, so
        # there is no bulk mean to clip toward — leave the spectrum intact
        # rather than invent one.
        return s
    clipped = eigvals.copy()
    clipped[bulk] = float(np.mean(eigvals[bulk]))
    return _rebuild(eigvecs, clipped)


def rotation_shrink(cov: Array, n_obs: int, alpha: float | None = None) -> Array:
    """Shrink sample eigenvalues toward the scaled-identity target ``mu * I``.

    ``S' = (1 - alpha) S + alpha * mu * I`` with ``mu = tr(S)/p`` — the
    eigenvectors (the "rotation") are kept and only the spectrum moves.
    ``alpha`` in ``[0, 1]`` may be given explicitly; when ``None`` the
    oracle-approximating intensity (Chen, Wiesel, Eldar & Hero 2010,
    eq. 23 large-p limit) is computed from the spectrum alone::

        alpha_hat = min(1, (p^2 tr(S^2) + p tr(S)^2)
                          / ((n + 1) (p^2 tr(S^2) - tr(S)^2)))

    which collapses to 1 exactly when the spectrum is already flat
    (``S = mu I``). Returns the rebuilt symmetric covariance as a
    ``np.float64`` array.
    """
    s = _check_cov(cov)
    n = _check_n_obs(n_obs)
    p = s.shape[0]
    tr_s = float(np.trace(s))
    if not np.isfinite(tr_s) or tr_s <= 0.0:
        raise ValueError("cov must have positive finite trace")
    mu = tr_s / p
    if alpha is None:
        a = _oas_alpha(s, n)
    else:
        a = float(alpha)
        if not np.isfinite(a) or not 0.0 <= a <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
    eigvals, eigvecs = _eigh(s)
    shrunk = (1.0 - a) * eigvals + a * mu
    return _rebuild(eigvecs, np.asarray(shrunk, dtype=np.float64))


def _sim_factor_state(
    n_obs: int,
    n_feat: int,
    k_factors: int,
    noise: float,
    seed: int,
) -> tuple[Array, Array]:
    """Draw the seeded factor world; returns ``(X, Sigma_true)``."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_obs, k_factors))
    loadings = rng.standard_normal((n_feat, k_factors))
    idio = rng.standard_normal((n_obs, n_feat))
    if k_factors > 0:
        # Row-normalize loadings so every feature's population variance is
        # exactly one: s^2 ||B_j||^2 = 1 - noise^2 for each j.
        row_norm = np.sqrt(np.sum(loadings * loadings, axis=1, keepdims=True))
        loadings = loadings / row_norm * math.sqrt(k_factors)
        scale = math.sqrt((1.0 - noise * noise) / k_factors)
    else:
        scale = 0.0
    lam = loadings * scale
    x = factors @ lam.T + noise * idio
    sigma_true = lam @ lam.T + noise * noise * np.eye(n_feat)
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(0.5 * (sigma_true + sigma_true.T), dtype=np.float64),
    )


def sim_correlated(
    n_obs: int,
    n_feat: int,
    k_factors: int,
    noise: float,
    seed: int,
) -> Array:
    """Seeded SYNTHETIC factor-structured Gaussian return panel.

    ``X = F (B s)^T + noise * Z`` with ``F`` ``(n_obs, k_factors)``,
    ``B`` ``(n_feat, k_factors)`` iid N(0,1) row-normalized to
    ``||B_j|| = sqrt(k_factors)``, ``Z`` ``(n_obs, n_feat)`` iid N(0,1) and
    ``s = sqrt((1 - noise^2) / k_factors)``, so each feature has exactly unit
    population variance and the population covariance is
    ``Sigma = s^2 B B^T + noise^2 I``: ``k_factors`` planted spikes on a
    ``noise^2`` MP bulk. ``noise`` is the idiosyncratic std dev in
    ``[0, 1]``; ``noise = 1`` (or ``k_factors = 0``) is the pure-noise null.
    All draws come from a single ``np.random.default_rng(seed)`` in a fixed
    order (factors, loadings, idiosyncratic), so identical seeds reproduce
    ``X`` bit-for-bit. Returns an ``(n_obs, n_feat)`` ``np.float64`` array.
    """
    n = _check_n_obs(n_obs)
    p = _check_int("n_feat", n_feat, 1)
    k = _check_int("k_factors", k_factors, 0)
    eps = float(noise)
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    x, _sigma_true = _sim_factor_state(n, p, k, eps, int(seed))
    return x


def bench_marchenko_pastur(seed: int = 0) -> dict[str, float]:
    """Float-only SYNTHETIC correctness bench for the MP denoising machinery.

    Two seeded worlds: a ``k=5``-factor world (``n_obs=250``, ``p=50``,
    ``q=0.2``, ``noise=0.6`` — planted spikes well above the MP edge) and a
    pure-noise world (``k=0``, ``noise=1``) for edge-violation rates. Keys
    are all ``synthetic_*`` floats; correctness evidence only, never market
    evidence. ``synthetic_determinism`` is 1.0 when two seeded runs of the
    bench's simulated panel agree bit-for-bit, else 0.0.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    s = int(seed)

    def fro_err(m: Array) -> float:
        return float(np.linalg.norm(m - sigma_sig, "fro"))

    # World 1: planted k-factor signal.
    n_obs, p, k, eps = _BENCH_N_OBS, _BENCH_N_FEAT, _BENCH_K_FACTORS, _BENCH_NOISE
    x_sig, sigma_sig = _sim_factor_state(n_obs, p, k, eps, s)
    cov_sig = np.asarray(x_sig.T @ x_sig / n_obs, dtype=np.float64)
    eig_sig = np.asarray(np.linalg.eigvalsh(cov_sig), dtype=np.float64)
    q = p / float(n_obs)
    edge = mp_edge_test(eig_sig, q, n_obs)
    detected = edge["count_above_edge"]
    detected_tw = edge["share_above_tw_edge"] * p

    clip = eigenvalue_clip(cov_sig, n_obs)
    shrunk = rotation_shrink(cov_sig, n_obs)
    err_raw = fro_err(cov_sig)
    err_clip = fro_err(clip)
    err_shrink = fro_err(shrunk)
    tr_sig = float(np.trace(cov_sig))

    # World 2: pure noise (Wishart null, Sigma = I).
    x_nil = sim_correlated(n_obs, p, 0, 1.0, s + 1)
    cov_nil = np.asarray(x_nil.T @ x_nil / n_obs, dtype=np.float64)
    eig_nil = np.asarray(np.linalg.eigvalsh(cov_nil), dtype=np.float64)
    lo_nil, hi_nil = mp_bounds(q, 1.0)
    edge_nil = mp_edge_test(eig_nil, q, n_obs, sigma2=1.0)
    viol = float(np.mean(((eig_nil < lo_nil) | (eig_nil > hi_nil)).astype(np.float64)))

    # Density mass check: q = 0.5 continuous density must integrate to 1.
    _, lam_max_half = mp_bounds(0.5, 1.0)
    grid = np.linspace(1e-12, lam_max_half, _BENCH_GRID)
    dens = mp_density(grid, 0.5)
    mass = float(np.trapezoid(dens, grid))

    det = float(np.array_equal(x_sig, sim_correlated(n_obs, p, k, eps, s)))

    out = {
        "synthetic_signal_count_err": float(abs(detected - k)),
        "synthetic_signal_count_err_tw": float(abs(detected_tw - k)),
        "synthetic_edge_share_signal": float(detected / p),
        "synthetic_edge_share_noise": edge_nil["share_above_edge"],
        "synthetic_tw_edge_viol_rate": edge_nil["share_above_tw_edge"],
        "synthetic_mp_bound_viol_rate": viol,
        "synthetic_clip_fro_improvement": (err_raw - err_clip) / err_raw,
        "synthetic_shrink_fro_improvement": (err_raw - err_shrink) / err_raw,
        "synthetic_clip_trace_err": abs(float(np.trace(clip)) - tr_sig) / tr_sig,
        "synthetic_shrink_trace_err": abs(float(np.trace(shrunk)) - tr_sig) / tr_sig,
        "synthetic_clip_min_eig": float(np.min(np.linalg.eigvalsh(clip))),
        "synthetic_oas_alpha": _oas_alpha(cov_sig, n_obs),
        "synthetic_density_mass_err": abs(mass - 1.0),
        "synthetic_determinism": det,
    }
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError("bench produced a non-finite diagnostic")
    return out
